"""
pipeline/safe_diffusion.py
──────────────────────────
End-to-end SafeDiffusion pipeline with multi-GPU support.

Device strategy
---------------
SD-1.x / SDXL  — single GPU (cuda:2 when ≥3 GPUs available, else cuda:0).
                  Auditor, inpainter, and policy share the same device.

SD-3.5          — split across two GPUs to avoid OOM on a single A6000 (48 GB):
                    cuda:0  SD3 transformer (MMDiT, ~18 GB fp16)
                    cuda:1  SD3 VAE + T5/CLIP text encoders (~12 GB fp16)
                    cuda:2  Auditor + Inpainter + Policy (~8 GB total)

  At audit boundaries the decoded PIL image (CPU numpy) is transferred between
  devices — this is zero-copy via PIL and costs only a small PCIe transfer for
  the 1024×1024 RGB image (~3 MB).  Latents are moved explicitly with .to()
  only when needed for VAE encode/decode.

  The SD3 VAE decode is called on base_device but the VAE lives on
  sd3_vae_device, so we move latents to sd3_vae_device for decode and move
  the result back to base_device for the next denoising step.

SD-1.x and SDXL are completely unaffected — their DevicePlan puts everything
on a single device identical to the previous single-GPU behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

from auditor.auditor import AdversarialAuditor
from inpainting.inpainter import build_inpainter, run_inpainting
from policy.tspo_policy import load_policy, load_state_encoder, get_knobs
from reinsertion.reinsertion import reinsert, decode_latents, pil_to_latent
from tournament.winner import select_winner
from utils.device_plan import make_device_plan, DevicePlan
from utils.diffusion_utils import encode_prompt, build_mask, noise_aware_heatmap
from utils.hf_auth import resolve_hf_token, check_gated


ModelFamily = Literal["sd1x", "sdxl", "sd3x"]


# ─────────────────────────────────────────────────────────────────────────────
# Family detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_family(model_id: str) -> ModelFamily:
    lower = model_id.lower()
    if any(k in lower for k in ("stable-diffusion-3", "sd3", "sd-3")):
        return "sd3x"
    if any(k in lower for k in ("xl", "sdxl")):
        return "sdxl"
    return "sd1x"


_NATIVE_RES: dict[ModelFamily, tuple[int, int]] = {
    "sd1x": (512, 512),
    "sdxl": (1024, 1024),
    "sd3x": (1024, 1024),
}


# ─────────────────────────────────────────────────────────────────────────────
# Base pipeline loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_sd1x(model_id, dtype, plan: DevicePlan, token=None):
    check_gated(model_id, token)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(plan.base_device)
    pipe.scheduler     = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    return pipe


def _load_sdxl(model_id, dtype, plan: DevicePlan, token=None):
    check_gated(model_id, token)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(plan.base_device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _load_sd3x(model_id, dtype, plan: DevicePlan, token=None):
    """
    Load SD3 with transformer and VAE/encoders on separate devices.

    diffusers StableDiffusion3Pipeline keeps all submodules together by
    default.  We load to CPU first, then move each submodule individually
    to avoid ever materialising the full model on one GPU.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=dtype, token=token,
        # Load to CPU first — we shard manually below
        device_map=None,
    )

    # Transformer (largest component) → base_device (cuda:0)
    pipe.transformer = pipe.transformer.to(plan.base_device)

    # VAE + text encoders → sd3_vae_device (cuda:1 when ≥3 GPUs)
    pipe.vae            = pipe.vae.to(plan.sd3_vae_device)
    pipe.text_encoder   = pipe.text_encoder.to(plan.sd3_vae_device)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(plan.sd3_vae_device)
    pipe.text_encoder_3 = pipe.text_encoder_3.to(plan.sd3_vae_device)

    # SD3.x always uses FlowMatchEulerDiscreteScheduler — rectified flow,
    # never EulerDiscreteScheduler (DDPM-style).  EulerDiscreteScheduler
    # applies a completely different noise parameterisation and will corrupt
    # the latent trajectory even for non-turbo checkpoints.
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # Enable cpu offload only as last resort (single GPU)
    if plan.use_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    return pipe


def _load_base_pipeline(model_id, dtype, plan: DevicePlan, token=None):
    family  = _detect_family(model_id)
    loaders = {"sd1x": _load_sd1x, "sdxl": _load_sdxl, "sd3x": _load_sd3x}
    pipe    = loaders[family](model_id, dtype, plan, token=token)
    print(f"  [Base] {model_id}  (family={family})")
    print(plan.summary())
    return pipe, family


# ─────────────────────────────────────────────────────────────────────────────
# Decode helpers (device-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_pil(pipe, latents: torch.Tensor, family: ModelFamily, plan: DevicePlan) -> Image.Image:
    """
    Decode latents → PIL.

    For SD3x the VAE lives on sd3_vae_device (possibly different from the
    transformer device).  We move latents there for decode, then bring the
    result back to CPU as a PIL image (PIL is always CPU / numpy).

    SD3.5 VAE formula (from diffusers StableDiffusion3Pipeline._decode):
        decoded = vae.decode(latents / scaling_factor + shift_factor)
    Both factors come from vae.config; hardcoding them is wrong because
    different SD3 checkpoints may differ.
    """
    if family == "sd3x":
        vae_dev = plan.sd3_vae_device
        lat     = latents.to(vae_dev)
        with torch.no_grad():
            # Exact formula from diffusers SD3 pipeline — order matters:
            #   unscale first (divide by scaling_factor), then shift
            scaling = pipe.vae.config.scaling_factor          # ≈ 1.5305
            shift   = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0  # ≈ 0.0609
            decoded_input = lat / scaling + shift
            out = pipe.vae.decode(decoded_input).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        arr = out[0].detach().permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    elif family == "sdxl":
        vae_dev = plan.base_device
        lat     = latents.to(vae_dev)
        with torch.no_grad():
            out = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        arr = out[0].detach().permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    else:
        return decode_latents(pipe, latents)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt encoding (device-aware for SD3x)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_prompt_for_pipe(pipe, prompt, plan: DevicePlan, family, dtype):
    """
    Encode prompt into embeddings.

    For SD3x the text encoders live on sd3_vae_device.  encode_prompt
    internally uses the encoder devices, so we pass sd3_vae_device as the
    device argument and then move the resulting embeddings to base_device
    for use in the transformer denoising loop.
    """
    if family == "sd1x":
        return encode_prompt(pipe, prompt, plan.base_device).to(dtype=dtype)

    if family == "sdxl":
        pe, npe, ppe, npp = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=plan.base_device,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt="", negative_prompt_2="",
        )
        return {
            "prompt_embeds":                pe.to(plan.base_device, dtype=dtype),
            "negative_prompt_embeds":       npe.to(plan.base_device, dtype=dtype),
            "pooled_prompt_embeds":         ppe.to(plan.base_device, dtype=dtype),
            "negative_pooled_prompt_embeds": npp.to(plan.base_device, dtype=dtype),
        }

    if family == "sd3x":
        # Text encoders live on sd3_vae_device — encode there, move to base_device
        pe, npe, ppe, npp = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            device=plan.sd3_vae_device,        # ← encoder device
            num_images_per_prompt=1, do_classifier_free_guidance=True,
        )
        return {
            # Move to transformer device for the denoising loop
            "prompt_embeds":                pe.to(plan.base_device, dtype=dtype),
            "negative_prompt_embeds":       npe.to(plan.base_device, dtype=dtype),
            "pooled_prompt_embeds":         ppe.to(plan.base_device, dtype=dtype),
            "negative_pooled_prompt_embeds": npp.to(plan.base_device, dtype=dtype),
        }

    raise ValueError(f"Unknown family: {family}")


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    image:         Image.Image
    prompt:        str
    interventions: int
    final_adv:     float
    final_safe:    bool
    trajectory:    list[dict] = field(default_factory=list)
    metrics:       dict       = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Tournament visualisation (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _save_tournament_figure(
    tournament_idx, step_idx, candidates, cand_scores,
    mask_pil, winner_pil, winner_res, best_idx, utilities, gate_infos,
    control_res, out_dir,
):
    os.makedirs(out_dir, exist_ok=True)
    n_cands  = len(candidates)
    mask_arr = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0

    def _overlay(img):
        rgb  = np.array(img.convert("RGB").resize((512, 512)), dtype=np.uint8)
        rgba = np.dstack([rgb, np.ones((512, 512), dtype=np.uint8) * 255])
        for c, val in enumerate([200, 0, 0]):
            rgba[..., c] = np.clip(
                rgba[..., c].astype(np.float32) * (1 - mask_arr * 0.45)
                + val * mask_arr * 0.45, 0, 255,
            ).astype(np.uint8)
        return rgba

    n_cols = n_cands + 1
    fig_w  = max(n_cols * 3.4, 9)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(fig_w, 7.2), dpi=120,
        gridspec_kw={"height_ratios": [3, 1.6]},
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.patch.set_facecolor("#0f0f1a")
    _G2, _R2, _N2 = "#44ff88", "#ff4444", "#aaaaaa"

    def _gc(ok): return _G2 if ok else _R2

    for col_idx in range(n_cols):
        ax = axes[0, col_idx]
        ax.set_facecolor("#0f0f1a"); ax.set_xticks([]); ax.set_yticks([])
        if col_idx < n_cands:
            util      = utilities[col_idx]
            is_winner = (col_idx == best_idx)
            ax.imshow(_overlay(candidates[col_idx]))
            bc = "#FFD700" if is_winner else "#555566"
            for sp in ax.spines.values():
                sp.set_edgecolor(bc); sp.set_linewidth(3.5 if is_winner else 1.0)
            ax.set_title(
                f"Candidate {col_idx}" + ("  ★ WINNER" if is_winner else ""),
                color="#FFD700" if is_winner else "#cccccc", fontsize=8,
                fontweight="bold" if is_winner else "normal", pad=4,
            )
            ax.text(0.5, -0.02, f"utility = {util:.4f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=7,
                    color="#FFD700" if is_winner else _N2,
                    fontweight="bold" if is_winner else "normal")
        else:
            w_rgb = np.array(winner_pil.convert("RGB").resize((512, 512)), dtype=np.uint8)
            ax.imshow(w_rgb)
            for sp in ax.spines.values():
                sp.set_edgecolor("#00e5ff"); sp.set_linewidth(3.5)
            kept = (best_idx == -1)
            ax.set_title("Control kept" if kept else f"Winner (cand {best_idx})",
                         color="#00e5ff", fontsize=8, fontweight="bold", pad=4)
            ax.text(0.5, -0.02,
                    f"adv={winner_res['adv_prob']:.3f}  policy={winner_res['policy_score']:.3f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#00e5ff")

    gi0 = gate_infos[0]
    for col_idx in range(n_cols):
        ax = axes[1, col_idx]
        ax.set_facecolor("#12122a"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355"); sp.set_linewidth(0.8)
        if col_idx < n_cands:
            gi, util = gate_infos[col_idx], utilities[col_idx]
            lines = [
                (f"─── gate debug  (cand {col_idx}) ───", "#8888bb", "bold"),
                (f"ctrl   policy={gi['ctrl_policy']:.3f}  adv={gi['ctrl_adv']:.3f}", "#6699cc", "normal"),
                (f"Δpolicy = {gi['raw_margin']:+.4f}  (>{gi['delta']})  " + ("PASS" if gi["margin_ok"] else "FAIL"), _gc(gi["margin_ok"]), "normal"),
                (f"policy  = {gi['policy']:.3f}  (≥{gi['tau_P']})  " + ("PASS" if gi["policy_ok"] else "FAIL"), _gc(gi["policy_ok"]), "normal"),
                (f"faith   = {gi['faith']:.3f}  (≥{gi['tau_F']})  " + ("PASS" if gi["faith_ok"] else "FAIL"), _gc(gi["faith_ok"]), "normal"),
                (f"adv     = {gi['adv']:.3f}  (<ctrl {gi['ctrl_adv']:.3f})  " + ("PASS" if gi["adv_ok"] else "FAIL"), _gc(gi["adv_ok"]), "normal"),
                (f"seam    = {gi['seam']:.3f}", _N2, "normal"),
                (f"utility = {util:.4f}" + ("  ★" if col_idx == best_idx and util > 0 else "") + ("  REJECTED" if util == 0 else ""),
                 "#FFD700" if (col_idx == best_idx and util > 0) else (_R2 if util == 0 else _G2), "bold"),
            ]
        else:
            kept = (best_idx == -1)
            lines = [
                ("─── final winner ───", "#8888bb", "bold"),
                ("source: CONTROL" if kept else f"source: cand {best_idx}", "#00e5ff", "bold"),
                (f"policy = {winner_res['policy_score']:.3f}", _N2, "normal"),
                (f"faith  = {winner_res['faithfulness']:.3f}", _N2, "normal"),
                (f"adv    = {winner_res['adv_prob']:.3f}", _N2, "normal"),
                (f"seam   = {winner_res['seam_quality']:.3f}", _N2, "normal"),
                (f"ctrl adv = {control_res['adv_prob']:.3f}", "#6699cc", "normal"),
                (f"Δadv = {control_res['adv_prob'] - winner_res['adv_prob']:+.4f}",
                 _gc(control_res["adv_prob"] > winner_res["adv_prob"]), "bold"),
            ]
        y = 0.97; dy = 1.0 / (len(lines) + 1)
        for text, color, weight in lines:
            ax.text(0.04, y, text, transform=ax.transAxes, ha="left", va="top",
                    fontsize=5.8, color=color, fontweight=weight, fontfamily="monospace")
            y -= dy

    fig.suptitle(
        f"Tournament #{tournament_idx + 1}  |  step {step_idx}  "
        f"|  {n_cands} candidate(s)  "
        f"|  δ={gi0['delta']}  τ_P={gi0['tau_P']}  τ_F={gi0['tau_F']}",
        color="white", fontsize=9, fontweight="bold", y=1.01,
    )
    plt.tight_layout(pad=0.6)
    fname = f"tournament_{tournament_idx + 1:03d}_step_{step_idx:03d}.png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Tournament vis] saved → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SafeDiffusionPipeline:
    """
    End-to-end SafeDiffusion pipeline.

    Multi-GPU behaviour (7× A6000)
    ──────────────────────────────
    SD-1.x / SDXL  → all models on cuda:2  (leaves cuda:0/1 free)
    SD-3.5         → transformer  cuda:0
                     VAE+encoders cuda:1
                     auditor/inpainter/policy cuda:2

    PIL images cross device boundaries as CPU numpy (via PIL) — zero GPU memory
    for the transfer itself, only a small PCIe copy of the RGB bytes.
    Latents are moved explicitly with .to() only at VAE encode/decode.

    SD-1.x and SDXL are completely unaffected — their plan puts everything on
    one device, identical to the previous single-GPU behaviour.
    """

    def __init__(self, cfg: dict, hf_token: str | None = None):
        self.cfg      = cfg
        n_gpus        = torch.cuda.device_count()
        self.hf_token = resolve_hf_token(hf_token)
        family = _detect_family(cfg["base_sd_model"])
        # SD3.5 MMDiT overflows in float16 — bfloat16 has same memory footprint
        # but 8 exponent bits (vs 5) so activation ranges don't overflow
        self.dtype = torch.bfloat16 if (family == "sd3x" and n_gpus > 0) else torch.bfloat16
        # Support models (inpainter, auditor) are always SD-1.x based — keep float16
        self.support_dtype = torch.bfloat16
        print(f"[SafeDiffusion] {n_gpus} CUDA device(s) detected.  base_dtype={self.dtype}  support_dtype={self.support_dtype}")
        print("[SafeDiffusion] Loading models …")
        self._load_models()
        print("[SafeDiffusion] Ready.")

    def _load_models(self):
        cfg    = self.cfg
        family = _detect_family(cfg["base_sd_model"])
        n_gpus = torch.cuda.device_count()

        # Build device plan — determines which GPU each component goes on
        self.plan: DevicePlan = make_device_plan(family, n_gpus)

        # Base generator
        self.pipe, self.base_family = _load_base_pipeline(
            cfg["base_sd_model"], self.dtype, self.plan, token=self.hf_token
        )
        self.base_res = _NATIVE_RES[self.base_family]

        # Support models (auditor, inpainter, policy) → support_device
        sdev = self.plan.support_device

        self.inpaint_pipe = build_inpainter(
            model_id   = cfg["inpainter_model"],
            lora_path  = cfg.get("inpainter_lora_path"),
            lora_scale = cfg.get("inpainter_lora_scale", 0.8),
            vae_from   = None,
            device     = sdev,
            dtype      = self.support_dtype,   # always float16 — inpainter is SD-1.x
        )
        print(f"  [Inpainter] {cfg['inpainter_model']}  device={sdev}")

        self.auditor = AdversarialAuditor(
            model_path = cfg["auditor_weights"],
            vocab_path = cfg["auditor_vocab"],
            device     = str(sdev),
        )
        print(f"  [Auditor] loaded  device={sdev}")

        if cfg.get("use_tspo", False):
            self.policy        = load_policy(cfg.get("tspo_checkpoint"), device=sdev)
            self.state_encoder = load_state_encoder(cfg.get("encoder_checkpoint"), device=sdev)
        else:
            self.policy        = None
            self.state_encoder = None
        print(f"  [Policy] {'TSPO active' if self.policy else 'Vanilla sweep mode'}  device={sdev}")

    def swap_base_model(self, model_id: str):
        """Hot-swap the base generator, rebuilding the device plan for the new family."""
        print(f"[SafeDiffusion] Swapping base model → {model_id}")
        family     = _detect_family(model_id)
        n_gpus     = torch.cuda.device_count()
        self.plan  = make_device_plan(family, n_gpus)
        self.pipe, self.base_family = _load_base_pipeline(
            model_id, self.dtype, self.plan, token=self.hf_token
        )
        self.base_res = _NATIVE_RES[self.base_family]

    def swap_inpainter_lora(self, lora_path: str, lora_scale: float = 0.8):
        from inpainting.inpainter import swap_lora
        swap_lora(self.inpaint_pipe, lora_path, lora_scale)

    # ── Denoising step ────────────────────────────────────────────────────────

    def _run_base_step(self, latents, t, text_emb, cfg_scale):
        """All tensors must already be on plan.base_device before calling."""
        family = self.base_family
        pipe   = self.pipe

        if family == "sd1x":
            li       = torch.cat([latents] * 2)
            li       = pipe.scheduler.scale_model_input(li, t)
            np_      = pipe.unet(li, t, encoder_hidden_states=text_emb).sample
            u_n, c_n = np_.chunk(2)
            return pipe.scheduler.step(u_n + cfg_scale * (c_n - u_n), t, latents).prev_sample

        if family == "sdxl":
            pe, npe  = text_emb["prompt_embeds"], text_emb["negative_prompt_embeds"]
            ppe, npp = text_emb["pooled_prompt_embeds"], text_emb["negative_pooled_prompt_embeds"]
            add_cond = {"text_embeds": ppe, "time_ids": self._xl_time_ids}
            add_unc  = {"text_embeds": npp, "time_ids": self._xl_time_ids}
            li       = torch.cat([latents] * 2)
            li       = pipe.scheduler.scale_model_input(li, t)
            enc      = torch.cat([npe, pe])
            add      = {k: torch.cat([u, c]) for (k, u), (_, c) in zip(add_unc.items(), add_cond.items())}
            np_      = pipe.unet(li, t, encoder_hidden_states=enc, added_cond_kwargs=add).sample
            u_n, c_n = np_.chunk(2)
            return pipe.scheduler.step(u_n + cfg_scale * (c_n - u_n), t, latents).prev_sample

        if family == "sd3x":
            pe, npe  = text_emb["prompt_embeds"], text_emb["negative_prompt_embeds"]
            ppe, npp = text_emb["pooled_prompt_embeds"], text_emb["negative_pooled_prompt_embeds"]
            li       = torch.cat([latents] * 2)
            if hasattr(pipe.scheduler, "scale_model_input"):
                li = pipe.scheduler.scale_model_input(li, t)
            enc, pool = torch.cat([npe, pe]), torch.cat([npp, ppe])
            # transformer is on base_device — all inputs already there
            np_      = pipe.transformer(
                li, timestep=t.expand(li.shape[0]),
                encoder_hidden_states=enc, pooled_projections=pool,
            ).sample
            u_n, c_n = np_.chunk(2)
            return pipe.scheduler.step(u_n + cfg_scale * (c_n - u_n), t, latents).prev_sample

        raise ValueError(f"Unknown family: {family}")

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, seed: int | None = None) -> GenerationResult:
        cfg    = self.cfg
        family = self.base_family
        plan   = self.plan
        H, W   = self.base_res
        seed   = seed if seed is not None else cfg.get("seed", 42)
        sdev   = plan.support_device   # auditor / inpainter / policy device

        method = cfg.get("reinsertion_method", "SD4_FLOW_INV")
        if family == "sd3x" and method not in ("SD0_DDPM", "SD4_FLOW_INV"):
            print(f"[SafeDiffusion] Forcing reinsertion_method=SD4_FLOW_INV for sd3x (flow matching)")
            method = "SD4_FLOW_INV"
        elif family == "sd3x" and method == "SD0_DDPM":
            print("[SafeDiffusion] WARNING: SD0_DDPM applies DDPM noise schedule which is incompatible "
                  "with FlowMatchEulerDiscrete — trajectory will be corrupted. Use SD4_FLOW_INV instead.")

        self.pipe.scheduler.set_timesteps(cfg["total_steps"], device=plan.base_device)
        timesteps = self.pipe.scheduler.timesteps
        audit_set = set(cfg["audit_steps"])

        gen = torch.Generator(device=plan.base_device).manual_seed(seed)

        if family == "sd1x":
            latent_c, latent_h, latent_w = 4, H // 8, W // 8
        elif family == "sdxl":
            latent_c, latent_h, latent_w = 4, H // 8, W // 8
            self._xl_time_ids = torch.tensor(
                [[H, W, 0, 0, H, W]], device=plan.base_device, dtype=self.dtype
            )
        elif family == "sd3x":
            latent_c, latent_h, latent_w = 16, H // 8, W // 8

        # Latents start on base_device (transformer/UNet device)
        init_sigma = getattr(self.pipe.scheduler, "init_noise_sigma", 1.0)
        latents = torch.randn(
            (1, latent_c, latent_h, latent_w),
            device=plan.base_device, dtype=self.dtype, generator=gen,
        ) * init_sigma

        # Prompt embeddings: encoded on encoder device, moved to base_device
        text_emb = _encode_prompt_for_pipe(self.pipe, prompt, plan, family, self.dtype)

        trajectory    = []
        first_pil     = None
        null_cache:   dict = {}
        interventions    = 0
        tournament_count = 0
        _tourn_dir = os.path.join(cfg.get("results_dir", "results"), "tournament_results")

        for i, t in enumerate(timesteps):
            with torch.no_grad():
                latents = self._run_base_step(latents, t, text_emb, cfg["guidance_scale"])

            # Clamp after every step — stops NaN propagating through fp16 UNet
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"  [UNet step {i}] WARNING: NaN/Inf — clamping")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

            if i not in audit_set:
                continue

            # ── Decode → PIL (crosses device boundary for SD3x) ───────────────
            # _decode_pil moves latents to vae_device internally for SD3x,
            # returns a CPU PIL image (no device dependency after this point)
            #
            # t_norm: SD1x/SDXL timesteps run 0→1000 (so divide by 1000.0).
            #         SD3x FlowMatchEulerDiscrete timesteps run 1→0 (already in [0,1]).
            #         We normalise by timesteps[0] (the max) to get [0,1] for all families.
            t_max  = float(timesteps[0].item())
            t_norm = float(t.item()) / (t_max if t_max > 0 else 1.0)
            current_pil = _decode_pil(self.pipe, latents, family, plan)

            arr_check = np.array(current_pil)
            if arr_check.mean() < 5.0:
                print(f"  [Step {i}] WARNING: decoded image mostly black — skipping audit")
                continue

            if first_pil is None:
                first_pil = current_pil

            # ── Audit (runs on support_device, input is CPU PIL) ──────────────
            # AdversarialAuditor.audit_pil() converts PIL → tensor internally
            # on its own device — no explicit transfer needed here.
            res_0 = self.auditor.audit_pil(current_pil, prompt, t_norm)

            hmap = res_0["heatmap"]
            if cfg.get("noise_aware_mask", False):
                hmap = noise_aware_heatmap(hmap, t, self.pipe.scheduler)
            mask_pil = build_mask(
                hmap,
                feather_sigma = cfg.get("blend_feather_sigma", 5),
                pct           = cfg.get("mask_percentile", 65),
            )

            step_rec = {
                "step": i, "t_norm": t_norm,
                "adv_before": res_0["adv_prob"], "adv_after": res_0["adv_prob"],
                "policy_before": res_0["policy_score"], "policy_after": res_0["policy_score"],
                "intervened": False, "u_max": 0.0,
            }

            class_triggered = res_0["harm_class"] in ("nudity", "violence")
            score_triggered = res_0["adv_prob"] >= 0.40
            if not (score_triggered or class_triggered):
                trajectory.append(step_rec)
                continue

            print(f"  [Step {i}] Violation ({res_0['harm_class']}, adv={res_0['adv_prob']:.3f}). Running tournament …")

            # ── TSPO knob generation (on support_device) ──────────────────────
            if self.policy is not None and self.state_encoder is not None:
                # img_embed and text_embed come from auditor (already on sdev)
                img_embed  = res_0["img_embed"].unsqueeze(0).to(sdev)
                text_embed = res_0["text_embed"].unsqueeze(0).to(sdev)
                mask_arr   = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
                mask_mean  = torch.tensor([[mask_arr.mean()]], device=sdev)
                # latents live on base_device — move a float copy to sdev for state encoder
                latent_b   = latents.detach().to(sdev, dtype=torch.float32)
                if latent_b.shape[1] != 4:
                    # TSPO StateEncoder was trained on 4-channel latents (SD-1.x/XL)
                    if latent_b.shape[1] % 4 == 0:
                        b, c, h, w = latent_b.shape
                        latent_b = latent_b.view(b, 4, c // 4, h, w).mean(dim=2)
                    else:
                        latent_b = latent_b[:, :4]
                t_t        = torch.tensor([[t_norm]], device=sdev)

                with torch.no_grad():
                    state_vec = self.state_encoder(text_embed, latent_b, img_embed, mask_mean, t_t)

                if torch.isnan(state_vec).any():
                    print("  [TSPO] WARNING: NaN in state_vec — sanitising and retrying")
                    latent_b  = torch.nan_to_num(latent_b,  nan=0.0, posinf=4.0, neginf=-4.0)
                    img_embed = torch.nan_to_num(img_embed, nan=0.0)
                    with torch.no_grad():
                        state_vec = self.state_encoder(text_embed, latent_b, img_embed, mask_mean, t_t)
                    if torch.isnan(state_vec).any():
                        print("  [TSPO] Retry NaN — falling back to vanilla sweep")
                        state_vec = None

                knobs = get_knobs(policy=self.policy, state=state_vec,
                                  n=cfg["n_candidates"], device=sdev)
            else:
                knobs = get_knobs(policy=None, state=None,
                                  n=cfg["n_candidates"], device=sdev)

            total_steps   = len(timesteps)
            steps_min     = cfg.get("inpaint_steps_min", 8)
            steps_max     = cfg.get("inpaint_steps_max", 30)
            inpaint_steps = max(steps_min, int((i / total_steps) * steps_max))
            print(f"  [Inpainter] step-scaled n_steps={inpaint_steps} "
                  f"(step {i}/{total_steps},  min={steps_min}  max={steps_max})")

            # ── Inpaint candidates (on support_device) ─────────────────────────
            # current_pil is CPU PIL — inpainter loads it onto sdev internally.
            # candidate PIL output is CPU PIL (returned from run_inpainting).
            candidates:  list[Image.Image] = []
            cand_scores: list[dict]        = []

            for knob in knobs:
                cand  = run_inpainting(
                    inpaint_pipe = self.inpaint_pipe,
                    base_pil     = current_pil,
                    mask_pil     = mask_pil,
                    harm_class   = res_0["harm_class"],
                    knob         = knob,
                    n_steps      = inpaint_steps,
                    device       = sdev,
                )
                res_c = self.auditor.audit_pil(cand, prompt, t_norm)
                candidates.append(cand)
                cand_scores.append(res_c)

            winner_pil, winner_res, best_idx, utilities, gate_infos, *_ = select_winner(
                candidates       = candidates,
                candidate_scores = cand_scores,
                control_pil      = current_pil,
                control_res      = res_0,
                cfg              = cfg,
            )

            _save_tournament_figure(
                tournament_idx = tournament_count,
                step_idx       = i,
                candidates     = candidates,
                cand_scores    = cand_scores,
                mask_pil       = mask_pil,
                winner_pil     = winner_pil,
                winner_res     = winner_res,
                best_idx       = best_idx,
                utilities      = utilities,
                gate_infos     = gate_infos,
                control_res    = res_0,
                out_dir        = _tourn_dir,
            )
            tournament_count += 1

            # ── Reinsert winner into latent trajectory ─────────────────────────
            # reinsert() encodes winner_pil using pipe.vae — for SD3x the VAE
            # lives on sd3_vae_device. _prepare_edit_latents in reinsertion.py
            # calls pil_to_latent(pipe, ...) which uses pipe.vae's device
            # automatically.  The returned latents are on vae_device; we move
            # them back to base_device before the next denoising step.
            if best_idx >= 0:
                latents = reinsert(
                    method       = method,
                    pipe         = self.pipe,
                    base_latents = latents,      # on base_device
                    winner_pil   = winner_pil,   # CPU PIL
                    mask_pil     = mask_pil,     # CPU PIL
                    t_norm       = t_norm,
                    t_idx        = i,
                    text_emb     = text_emb,     # on base_device
                    cfg          = cfg,
                    null_cache   = null_cache,
                )

                # Ensure latents are back on base_device after reinsert
                # (for SD3x the VAE encode/decode may return on sd3_vae_device)
                latents = latents.to(plan.base_device)

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print("  [Reinsert] WARNING: NaN/Inf — clamping")
                latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

                interventions += 1
                step_rec["intervened"] = True
                step_rec["u_max"]      = utilities[best_idx]
                res_after = self.auditor.audit_pil(
                    _decode_pil(self.pipe, latents, family, plan), prompt, t_norm
                )
                step_rec["adv_after"]    = res_after["adv_prob"]
                step_rec["policy_after"] = res_after["policy_score"]

            trajectory.append(step_rec)

        # ── Final decode + metrics ─────────────────────────────────────────────
        final_pil = _decode_pil(self.pipe, latents, family, plan)
        final_res = self.auditor.audit_pil(final_pil, prompt, 0.0)
        ctrl_res  = self.auditor.audit_pil(first_pil or final_pil, prompt, 0.0)

        metrics = {
            "prompt":          prompt,
            "method":          method,
            "base_family":     family,
            "interventions":   interventions,
            "adv_final":       final_res["adv_prob"],
            "adv_ctrl":        ctrl_res["adv_prob"],
            "adv_improvement": ctrl_res["adv_prob"] - final_res["adv_prob"],
            "faithfulness":    final_res["faithfulness"],
            "seam_quality":    final_res["seam_quality"],
        }

        if cfg.get("save_images", False):
            os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)
            slug     = prompt[:40].replace(" ", "_").replace("/", "")
            out_path = os.path.join(cfg.get("results_dir", "results"), f"{slug}_safe.png")
            final_pil.save(out_path)
            metrics["saved_to"] = out_path

        return GenerationResult(
            image         = final_pil,
            prompt        = prompt,
            interventions = interventions,
            final_adv     = final_res["adv_prob"],
            final_safe    = not final_res["is_unsafe"],
            trajectory    = trajectory,
            metrics       = metrics,
        )
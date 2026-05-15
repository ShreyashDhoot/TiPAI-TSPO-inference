from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
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
from utils.diffusion_utils import encode_prompt, build_mask, noise_aware_heatmap
from utils.hf_auth import resolve_hf_token, check_gated


ModelFamily = Literal["sd1x", "sdxl", "sd3x", "flux"]


def _detect_family(model_id: str) -> ModelFamily:
    lower = model_id.lower()
    if any(k in lower for k in ("flux", "flux.1")):
        return "flux"
    if any(k in lower for k in ("stable-diffusion-3", "sd3", "sd-3")):
        return "sd3x"
    if any(k in lower for k in ("xl", "sdxl")):
        return "sdxl"
    return "sd1x"


_NATIVE_RES: dict[ModelFamily, tuple[int, int]] = {
    "sd1x": (512, 512),
    "sdxl": (1024, 1024),
    "sd3x": (1024, 1024),
    "flux": (1024, 1024),
}


def _load_sd1x(model_id, dtype, device, token=None):
    check_gated(model_id, token)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    return pipe


def _load_sdxl(model_id, dtype, device, token=None):
    check_gated(model_id, token)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    return pipe


def _load_sd3x(model_id, dtype, device, token=None):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=dtype, token=token
    ).to(device)
    if "turbo" in model_id.lower():
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def _load_flux(model_id, dtype, device, token=None):
    from diffusers import FluxPipeline
    import torch
    check_gated(model_id, token)
    pipe = FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, token=token
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe


def _load_base_pipeline(model_id, dtype, device, token=None):
    family  = _detect_family(model_id)
    loaders = {"sd1x": _load_sd1x, "sdxl": _load_sdxl, "sd3x": _load_sd3x, "flux": _load_flux}
    pipe    = loaders[family](model_id, dtype, device, token=token)
    print(f"  [Base] {model_id}  (family={family})")
    return pipe, family


def _decode_pil(pipe, latents: torch.Tensor, family: ModelFamily, H: int = None, W: int = None) -> Image.Image:
    """Decode VAE latents to PIL. Clamps arr to [0,1] before uint8 cast to prevent
    NaN → black-pixel corruption that would propagate into the auditor embeddings."""
    if family == "flux":
        if H is None or W is None:
            raise ValueError("H and W must be provided for flux _decode_pil")
        latents_up = pipe._unpack_latents(latents, H, W, pipe.vae_scale_factor)
        latents_up = (latents_up / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        out = pipe.vae.decode(latents_up).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        arr = out[0].detach().permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))
    elif family in ("sd3x", "sdxl"):
        out = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        out = (out / 2 + 0.5).clamp(0, 1)
        # FIX: clamp float arr before cast — prevents RuntimeWarning + NaN pixels
        arr = out[0].detach().permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))
    else:
        return decode_latents(pipe, latents)


def _encode_prompt_for_pipe(pipe, prompt, device, family, dtype):
    if family == "sd1x":
        return encode_prompt(pipe, prompt, device).to(dtype=dtype)
    if family == "sdxl":
        pe, npe, ppe, npp = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=True,
            negative_prompt="", negative_prompt_2="",
        )
        return {
            "prompt_embeds": pe.to(dtype),
            "negative_prompt_embeds": npe.to(dtype),
            "pooled_prompt_embeds": ppe.to(dtype),
            "negative_pooled_prompt_embeds": npp.to(dtype),
        }
    if family == "sd3x":
        pe, npe, ppe, npp = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
        )
        return {
            "prompt_embeds": pe.to(dtype),
            "negative_prompt_embeds": npe.to(dtype),
            "pooled_prompt_embeds": ppe.to(dtype),
            "negative_pooled_prompt_embeds": npp.to(dtype),
        }
    if family == "flux":
        pe, ppe, txt_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, device=device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
        return {
            "prompt_embeds": pe.to(dtype),
            "pooled_prompt_embeds": ppe.to(dtype),
            "text_ids": txt_ids.to(dtype),
        }
    raise ValueError(f"Unknown family: {family}")


@dataclass
class GenerationResult:
    image:         Image.Image
    prompt:        str
    interventions: int
    final_adv:     float
    final_safe:    bool
    trajectory:    list[dict] = field(default_factory=list)
    metrics:       dict       = field(default_factory=dict)


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
            sc, util  = cand_scores[col_idx], utilities[col_idx]
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
            gi, sc, util = gate_infos[col_idx], cand_scores[col_idx], utilities[col_idx]
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


class SafeDiffusionPipeline:

    def __init__(self, cfg: dict, hf_token: str | None = None):
        self.cfg      = cfg
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype    = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.hf_token = resolve_hf_token(hf_token)
        print("[SafeDiffusion] Loading models …")
        self._load_models()
        print("[SafeDiffusion] Ready.")

    def _load_models(self):
        cfg = self.cfg
        self.pipe, self.base_family = _load_base_pipeline(
            cfg["base_sd_model"], self.dtype, self.device, token=self.hf_token
        )
        self.base_res = _NATIVE_RES[self.base_family]

        self.inpaint_pipe = build_inpainter(
            model_id   = cfg["inpainter_model"],
            lora_path  = cfg.get("inpainter_lora_path"),
            lora_scale = cfg.get("inpainter_lora_scale", 0.8),
            vae_from   = None,
            device     = self.device,
            dtype      = self.dtype,
        )
        print(f"  [Inpainter] {cfg['inpainter_model']}  (independent VAE)")

        self.auditor = AdversarialAuditor(
            model_path = cfg["auditor_weights"],
            vocab_path = cfg["auditor_vocab"],
            device     = "auto",
        )
        print("  [Auditor] loaded.")

        if cfg.get("use_tspo", False):
            self.policy        = load_policy(cfg.get("tspo_checkpoint"), device=self.device)
            self.state_encoder = load_state_encoder(cfg.get("encoder_checkpoint"), device=self.device)
        else:
            self.policy        = None
            self.state_encoder = None
        print(f"  [Policy] {'TSPO active' if self.policy else 'Vanilla sweep mode'}")

    def swap_base_model(self, model_id: str):
        print(f"[SafeDiffusion] Swapping base model → {model_id}")
        self.pipe, self.base_family = _load_base_pipeline(model_id, self.dtype, self.device, token=self.hf_token)
        self.base_res = _NATIVE_RES[self.base_family]

    def swap_inpainter_lora(self, lora_path: str, lora_scale: float = 0.8):
        from inpainting.inpainter import swap_lora
        swap_lora(self.inpaint_pipe, lora_path, lora_scale)

    def _run_base_step(self, latents, t, text_emb, cfg_scale, latent_image_ids=None):
        family = self.base_family
        pipe   = self.pipe

        if family == "sd1x":
            li      = torch.cat([latents] * 2)
            li      = pipe.scheduler.scale_model_input(li, t)
            np_     = pipe.unet(li, t, encoder_hidden_states=text_emb).sample
            u_n, c_n = np_.chunk(2)
            guided  = u_n + cfg_scale * (c_n - u_n)
            return pipe.scheduler.step(guided, t, latents).prev_sample

        if family == "sdxl":
            pe, npe = text_emb["prompt_embeds"], text_emb["negative_prompt_embeds"]
            ppe, npp = text_emb["pooled_prompt_embeds"], text_emb["negative_pooled_prompt_embeds"]
            add_cond = {"text_embeds": ppe, "time_ids": self._xl_time_ids}
            add_unc  = {"text_embeds": npp, "time_ids": self._xl_time_ids}
            li       = torch.cat([latents] * 2)
            li       = pipe.scheduler.scale_model_input(li, t)
            enc      = torch.cat([npe, pe])
            add      = {k: torch.cat([v_u, v_c]) for (k, v_u), (_, v_c) in zip(add_unc.items(), add_cond.items())}
            np_      = pipe.unet(li, t, encoder_hidden_states=enc, added_cond_kwargs=add).sample
            u_n, c_n = np_.chunk(2)
            guided   = u_n + cfg_scale * (c_n - u_n)
            return pipe.scheduler.step(guided, t, latents).prev_sample

        if family == "sd3x":
            pe, npe = text_emb["prompt_embeds"], text_emb["negative_prompt_embeds"]
            ppe, npp = text_emb["pooled_prompt_embeds"], text_emb["negative_pooled_prompt_embeds"]
            li       = torch.cat([latents] * 2)
            li       = pipe.scheduler.scale_model_input(li, t)
            enc, pool = torch.cat([npe, pe]), torch.cat([npp, ppe])
            np_      = pipe.transformer(li, timestep=t.expand(li.shape[0]),
                                        encoder_hidden_states=enc, pooled_projections=pool).sample
            u_n, c_n = np_.chunk(2)
            guided   = u_n + cfg_scale * (c_n - u_n)
            return pipe.scheduler.step(guided, t, latents).prev_sample

        if family == "flux":
            pe = text_emb["prompt_embeds"]
            ppe = text_emb["pooled_prompt_embeds"]
            text_ids = text_emb["text_ids"]
            has_guidance = getattr(pipe.transformer.config, "guidance_embeds", False)
            guidance = torch.full([1], cfg_scale, device=self.device, dtype=self.dtype).expand(latents.shape[0]) if has_guidance else None
            np_ = pipe.transformer(
                hidden_states=latents,
                timestep=(t / 1000.0).expand(latents.shape[0]),
                guidance=guidance,
                pooled_projections=ppe,
                encoder_hidden_states=pe,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            return pipe.scheduler.step(np_, t, latents, return_dict=False)[0]

        raise ValueError(f"Unknown family: {family}")

    def generate(self, prompt: str, seed: int | None = None) -> GenerationResult:
        cfg    = self.cfg
        family = self.base_family
        H, W   = self.base_res
        seed   = seed if seed is not None else cfg.get("seed", 42)

        # Reinsertion method
        method = cfg.get("reinsertion_method", "SD3_NULL_TEXT")
        if family in ("sd3x", "flux") and method != "SD0_DDPM":
            print(f"[SafeDiffusion] Forcing reinsertion_method=SD0_DDPM for family={family}")
            method = "SD0_DDPM"

        gen = torch.Generator(device=self.device).manual_seed(seed)
        latent_image_ids = None

        if family == "flux":
            sigmas = np.linspace(1.0, 1 / cfg["total_steps"], cfg["total_steps"])
            num_channels_latents = self.pipe.transformer.config.in_channels // 4
            latents, latent_image_ids = self.pipe.prepare_latents(
                batch_size=1, num_channels_latents=num_channels_latents,
                height=H, width=W, dtype=self.dtype, device=self.device,
                generator=gen, latents=None,
            )
            image_seq_len = latents.shape[1]
            from diffusers.pipelines.flux.pipeline_flux import calculate_shift
            mu = calculate_shift(
                image_seq_len,
                self.pipe.scheduler.config.get("base_image_seq_len", 256),
                self.pipe.scheduler.config.get("max_image_seq_len", 4096),
                self.pipe.scheduler.config.get("base_shift", 0.5),
                self.pipe.scheduler.config.get("max_shift", 1.15),
            )
            self.pipe.scheduler.set_timesteps(sigmas=sigmas, device=self.device, mu=mu)
        else:
            self.pipe.scheduler.set_timesteps(cfg["total_steps"], device=self.device)
            
        timesteps = self.pipe.scheduler.timesteps
        audit_set = set(cfg["audit_steps"])

        gen = torch.Generator(device=self.device).manual_seed(seed)

        if family == "sd1x":
            latent_c, latent_h, latent_w = 4, H // 8, W // 8
        elif family == "sdxl":
            latent_c, latent_h, latent_w = 4, H // 8, W // 8
            self._xl_time_ids = torch.tensor(
                [[H, W, 0, 0, H, W]], device=self.device, dtype=self.dtype
            )
        elif family == "sd3x":
            latent_c, latent_h, latent_w = 16, H // 8, W // 8

        if family != "flux":
            latents = torch.randn(
                (1, latent_c, latent_h, latent_w),
                device=self.device, dtype=self.dtype, generator=gen,
            ) * self.pipe.scheduler.init_noise_sigma

        text_emb = _encode_prompt_for_pipe(self.pipe, prompt, self.device, family, self.dtype)

        trajectory    = []
        first_pil     = None
        null_cache:   dict = {}
        interventions    = 0
        tournament_count = 0
        _tourn_dir = os.path.join(cfg.get("results_dir", "results"), "tournament_results")

        for i, t in enumerate(timesteps):
            with torch.no_grad():
                latents = self._run_base_step(latents, t, text_emb, cfg["guidance_scale"], latent_image_ids=latent_image_ids)

            # FIX 1: clamp latents after every UNet step to stop NaN propagation
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"  [UNet step {i}] WARNING: NaN/Inf in latents — clamping")
            latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

            if i not in audit_set:
                continue

            t_norm      = float(t.item()) / 1000.0
            current_pil = _decode_pil(self.pipe, latents, family, H=H, W=W)

            # FIX 2: skip audit if decoded image is mostly black (NaN-origin artifact)
            arr_check = np.array(current_pil)
            if arr_check.mean() < 5.0:
                print(f"  [Step {i}] WARNING: decoded image mostly black (mean={arr_check.mean():.1f}) — skipping audit")
                continue

            if first_pil is None:
                first_pil = current_pil

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

            # TSPO knob generation
            if self.policy is not None and self.state_encoder is not None:
                img_embed  = res_0["img_embed"].unsqueeze(0).to(self.device)
                text_embed = res_0["text_embed"].unsqueeze(0).to(self.device)
                mask_arr   = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
                mask_mean  = torch.tensor([[mask_arr.mean()]], device=self.device)
                latent_b   = latents.to(self.device).float()
                t_t        = torch.tensor([[t_norm]], device=self.device)

                with torch.no_grad():
                    state_vec = self.state_encoder(text_embed, latent_b, img_embed, mask_mean, t_t)

                # FIX 3: guard NaN state_vec — clamp inputs and retry once, then fall back
                if torch.isnan(state_vec).any():
                    print("  [TSPO] WARNING: NaN in state_vec — sanitising inputs and retrying")
                    latent_b  = torch.nan_to_num(latent_b,  nan=0.0, posinf=4.0, neginf=-4.0)
                    img_embed = torch.nan_to_num(img_embed, nan=0.0)
                    with torch.no_grad():
                        state_vec = self.state_encoder(text_embed, latent_b, img_embed, mask_mean, t_t)
                    if torch.isnan(state_vec).any():
                        print("  [TSPO] Retry also NaN — falling back to vanilla sweep")
                        state_vec = None

                knobs = get_knobs(policy=self.policy, state=state_vec,
                                  n=cfg["n_candidates"], device=self.device)
            else:
                knobs = get_knobs(policy=None, state=None,
                                  n=cfg["n_candidates"], device=self.device)

            total_steps   = len(timesteps)
            steps_min     = cfg.get("inpaint_steps_min", 8)
            steps_max     = cfg.get("inpaint_steps_max", 30)
            inpaint_steps = max(steps_min, int((i / total_steps) * steps_max))
            print(f"  [Inpainter] step-scaled n_steps={inpaint_steps} "
                  f"(step {i}/{total_steps},  min={steps_min}  max={steps_max})")

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
                    device       = self.device,
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

            if best_idx >= 0:
                cfg_pass = cfg.copy()
                cfg_pass["H"] = H
                cfg_pass["W"] = W
                latents = reinsert(
                    method       = method,
                    pipe         = self.pipe,
                    base_latents = latents,
                    winner_pil   = winner_pil,   # _prepare_edit_latents in reinsert handles resize
                    mask_pil     = mask_pil,
                    t_norm       = t_norm,
                    t_idx        = i,
                    text_emb     = text_emb,      # passed for all families; reinsert uses if needed
                    cfg          = cfg_pass,
                    null_cache   = null_cache,
                )

                # FIX 4: clamp reinserted latents before next UNet step
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print("  [Reinsert] WARNING: NaN/Inf in reinserted latents — clamping")
                latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

                interventions += 1
                step_rec["intervened"] = True
                step_rec["u_max"]      = utilities[best_idx]
                res_after = self.auditor.audit_pil(
                    _decode_pil(self.pipe, latents, family, H=H, W=W), prompt, t_norm
                )
                step_rec["adv_after"]    = res_after["adv_prob"]
                step_rec["policy_after"] = res_after["policy_score"]

            trajectory.append(step_rec)

        final_pil = _decode_pil(self.pipe, latents, family, H=H, W=W)
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

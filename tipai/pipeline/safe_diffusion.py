"""
pipeline/safe_diffusion.py
──────────────────────────
End-to-end SafeDiffusion pipeline.

Send in a prompt → get back a safe PIL image.

Usage
-----
    from pipeline.safe_diffusion import SafeDiffusionPipeline
    from utils.config_loader import load_config

    cfg = load_config("config.yaml")
    sdp = SafeDiffusionPipeline(cfg)
    image = sdp.generate("a woman at the beach")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
from PIL import Image

from auditor.auditor import AdversarialAuditor
from inpainting.inpainter import build_inpainter, run_inpainting
from policy.tspo_policy import load_policy, load_state_encoder, get_knobs
from reinsertion.reinsertion import reinsert, decode_latents, pil_to_latent
from tournament.winner import select_winner
from utils.diffusion_utils import encode_prompt, build_mask, noise_aware_heatmap


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
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SafeDiffusionPipeline:
    """
    Wraps the full TiPAI-TSPO speculative-decoding loop.

    Parameters
    ----------
    cfg : dict loaded from config.yaml  (use utils.config_loader.load_config)
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch.float16 if self.device.type == "cuda" else torch.float32

        print("[SafeDiffusion] Loading models …")
        self._load_models()
        print("[SafeDiffusion] Ready.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self):
        cfg = self.cfg

        # 1. Base SD pipeline ─────────────────────────────────────────────────
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     cfg["base_sd_model"], torch_dtype=self.dtype, use_safetensors=True
        # ).to(self.device)
        is_sdxl = cfg.get("base_sd_model", "sd1") == "sdxl"

        if is_sdxl:
            self.pipe = DiffusionPipeline.from_pretrained(
                cfg["base_sd_model"], torch_dtype=self.dtype, use_safetensors=True
            ).to(self.device)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                cfg["base_sd_model"], torch_dtype=self.dtype, use_safetensors=True
            ).to(self.device)

        self.is_sdxl = is_sdxl
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        print(f"  [Base] {cfg['base_sd_model']}")

        # 2. Inpainter ────────────────────────────────────────────────────────
        self.inpaint_pipe = build_inpainter(
            model_id   = cfg["inpainter_model"],
            lora_path  = cfg.get("inpainter_lora_path"),
            lora_scale = cfg.get("inpainter_lora_scale", 0.8),
            vae_from   = self.pipe.vae,
            device     = self.device,
            dtype      = self.dtype,
        )
        print(f"  [Inpainter] {cfg['inpainter_model']}")

        # 3. Auditor ──────────────────────────────────────────────────────────
        self.auditor = AdversarialAuditor(
            model_path = cfg["auditor_weights"],
            vocab_path = cfg["auditor_vocab"],
            device     = "auto",
        )
        print("  [Auditor] loaded.")

        # 4. TSPO policy + StateEncoder ───────────────────────────────────────
        # tspo_checkpoint    → policy MLP weights  (raw state_dict .pth)
        # encoder_checkpoint → StateEncoder weights (separate file saved by
        #                       training as state_enc_step{N:05d}.pth)
        if cfg.get("use_tspo", False):
            self.policy        = load_policy(
                cfg.get("tspo_checkpoint"), device=self.device
            )
            self.state_encoder = load_state_encoder(
                cfg.get("encoder_checkpoint"), device=self.device
            )
        else:
            self.policy        = None
            self.state_encoder = None

        print(f"  [Policy] {'TSPO active' if self.policy else 'Vanilla sweep mode'}")

    # ── Public API ────────────────────────────────────────────────────────────

    def swap_base_model(self, model_id: str):
        """Hot-swap the base SD model."""
        print(f"[SafeDiffusion] Swapping base model → {model_id}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, use_safetensors=True
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        self.inpaint_pipe.vae = self.pipe.vae

    def swap_inpainter_lora(self, lora_path: str, lora_scale: float = 0.8):
        """Hot-swap the inpainter LoRA weights without reloading the whole model."""
        from inpainting.inpainter import swap_lora
        swap_lora(self.inpaint_pipe, lora_path, lora_scale)

    def generate(self, prompt: str, seed: int | None = None) -> GenerationResult:
        """
        Generate a safe image for the given prompt.

        Parameters
        ----------
        prompt : text prompt
        seed   : optional RNG seed (overrides config)
        """
        cfg    = self.cfg
        method = cfg.get("reinsertion_method", "SD3_NULL_TEXT")
        seed   = seed if seed is not None else cfg.get("seed", 42)

        # ── setup ─────────────────────────────────────────────────────────────
        self.pipe.scheduler.set_timesteps(cfg["total_steps"], device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        audit_set = set(cfg["audit_steps"])

        gen     = torch.Generator(device=self.device).manual_seed(seed)
        # latents = torch.randn(
        #     (1, 4, 64, 64), device=self.device, dtype=self.dtype, generator=gen
        # )

        latent_channels = 4
        latent_res = 128 if self.is_sdxl else 64  # SDXL 1024px → 128x128 latent

        latents = torch.randn(
            (1, latent_channels, latent_res, latent_res),
            device=self.device, dtype=self.dtype, generator=gen
)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # text_emb = encode_prompt(self.pipe, prompt, self.device).to(dtype=self.dtype)

        if self.is_sdxl:
    # SDXL needs both sequence embeddings AND pooled embeddings
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_embeds = \
                self.pipe.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
            text_emb = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_emb = torch.cat([negative_pooled_embeds, pooled_prompt_embeds])
        else:
            text_emb = encode_prompt(self.pipe, prompt, self.device).to(dtype=self.dtype)
            pooled_emb = None

        trajectory    = []
        first_pil     = None
        null_cache: dict = {}
        interventions = 0

        # ── main denoising loop ───────────────────────────────────────────────
        for i, t in enumerate(timesteps):
            with torch.no_grad():
                if i in null_cache:
                    current_emb = torch.cat([null_cache[i], text_emb[-1:]])
                else:
                    current_emb = text_emb

                li  = torch.cat([latents] * 2)
                li  = self.pipe.scheduler.scale_model_input(li, t)
                # np_ = self.pipe.unet(li, t, encoder_hidden_states=current_emb).sample
                if self.is_sdxl:
    # SDXL UNet also needs added_cond_kwargs with pooled embeddings + resolution
                    add_text_embeds = pooled_emb
                    add_time_ids = torch.tensor(
                        [[1024, 1024, 0, 0, 1024, 1024]] * 2,  # orig_size, crop_coords, target_size
                        device=self.device, dtype=self.dtype
                    )
                    np_ = self.pipe.unet(
                        li, t,
                        encoder_hidden_states=current_emb,
                        added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    ).sample
                else:
                    np_ = self.pipe.unet(li, t, encoder_hidden_states=current_emb).sample
                u_n, c_n = np_.chunk(2)
                np_     = u_n + cfg["guidance_scale"] * (c_n - u_n)
                latents = self.pipe.scheduler.step(np_, t, latents).prev_sample

            if i not in audit_set:
                continue

            # ── decode + audit ────────────────────────────────────────────────
            t_norm      = float(t.item()) / 1000.0
            current_pil = decode_latents(self.pipe, latents)
            if first_pil is None:
                first_pil = current_pil

            # audit_pil now returns img_embed (256,) and text_embed (512,)
            # alongside the standard fields — no CLIP needed
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

            # ── skip if safe ──────────────────────────────────────────────────
            class_triggered = res_0["harm_class"] in ("nudity", "violence")
            score_triggered = res_0["adv_prob"] >= 0.40
            if not (score_triggered or class_triggered):
                trajectory.append(step_rec)
                continue

            print(
                f"  [Step {i}] Violation ({res_0['harm_class']}, "
                f"adv={res_0['adv_prob']:.3f}). Running tournament …"
            )

            # ── TSPO knob generation ──────────────────────────────────────────
            if self.policy is not None and self.state_encoder is not None:
                # All inputs come from the auditor — no CLIP, no extra models.

                # img_embed  : (256,) from audit_pil  → unsqueeze to (1, 256)
                # text_embed : (512,) from audit_pil  → unsqueeze to (1, 512)
                img_embed  = res_0["img_embed"].unsqueeze(0).to(self.device)   # (1, 256)
                text_embed = res_0["text_embed"].unsqueeze(0).to(self.device)  # (1, 512)

                # mask_mean : scalar coverage of the inpaint region
                mask_arr  = np.array(mask_pil.convert("L"), dtype=np.float32) / 255.0
                mask_mean = torch.tensor([[mask_arr.mean()]], device=self.device)  # (1, 1)

                # latent and timestep
                latent_b = latents.to(self.device).float()                     # (1, 4, 64, 64)
                t_t      = torch.tensor([[t_norm]], device=self.device)        # (1, 1)

                with torch.no_grad():
                    state_vec = self.state_encoder(
                        text_embed,   # (1, 512)
                        latent_b,     # (1, 4, 64, 64)
                        img_embed,    # (1, 256)
                        mask_mean,    # (1, 1)
                        t_t,          # (1, 1)
                    )  # → (1, 257)

                knobs = get_knobs(
                    policy = self.policy,
                    state  = state_vec,
                    n      = cfg["n_candidates"],
                    device = self.device,
                )
            else:
                # Vanilla sweep — policy or encoder not loaded
                knobs = get_knobs(
                    policy = None,
                    state  = None,
                    n      = cfg["n_candidates"],
                    device = self.device,
                )

            # ── generate candidates ───────────────────────────────────────────
            candidates:  list[Image.Image] = []
            cand_scores: list[dict]        = []

            for knob in knobs:
                # cand = run_inpainting(
                #     inpaint_pipe  = self.inpaint_pipe,
                #     base_pil      = current_pil,
                #     mask_pil      = mask_pil,
                #     prompt        = prompt,
                #     harm_class    = res_0["harm_class"],
                #     knob          = knob,
                #     n_steps       = cfg.get("inpaint_steps", 20),
                #     prompt_mode   = cfg.get("inpaint_prompt_mode", "safe"),
                #     device        = self.device,
                # )
                if self.is_sdxl:
    # Decode gives 1024px; SD1.5 inpainter needs 512px
                    base_512  = current_pil.resize((512, 512), Image.LANCZOS)
                    mask_512  = mask_pil.resize((512, 512),    Image.NEAREST)
                else:
                    base_512  = current_pil
                    mask_512  = mask_pil

                cand = run_inpainting(
                    inpaint_pipe  = self.inpaint_pipe,
                    base_pil=base_512,
                    mask_pil=mask_512,
                    prompt        = prompt,
                    harm_class    = res_0["harm_class"],
                    knob          = knob,
                    n_steps       = cfg.get("inpaint_steps", 20),
                    prompt_mode   = cfg.get("inpaint_prompt_mode", "safe"),
                    device        = self.device,
                )

                if self.is_sdxl:
                    # Resize inpainted result back up before re-encoding into SDXL latent space
                    cand = cand.resize((1024, 1024), Image.LANCZOS)
                res_c = self.auditor.audit_pil(cand, prompt, t_norm)
                candidates.append(cand)
                cand_scores.append(res_c)

            # ── tournament winner selection ────────────────────────────────────
            winner_pil, winner_res, best_idx, utilities = select_winner(
                candidates       = candidates,
                candidate_scores = cand_scores,
                control_pil      = current_pil,
                control_res      = res_0,
                cfg              = cfg,
            )

            # ── reinsert winner into trajectory ───────────────────────────────
            if best_idx >= 0:
                latents = reinsert(
                    method       = method,
                    pipe         = self.pipe,
                    base_latents = latents,
                    winner_pil   = winner_pil,
                    mask_pil     = mask_pil,
                    t_norm       = t_norm,
                    t_idx        = i,
                    text_emb     = text_emb,
                    cfg          = cfg,
                    null_cache   = null_cache,
                )
                interventions += 1
                step_rec["intervened"] = True
                step_rec["u_max"]      = utilities[best_idx]
                res_after = self.auditor.audit_pil(
                    decode_latents(self.pipe, latents), prompt, t_norm
                )
                step_rec["adv_after"]    = res_after["adv_prob"]
                step_rec["policy_after"] = res_after["policy_score"]

            trajectory.append(step_rec)

        # ── final decode + metrics ────────────────────────────────────────────
        final_pil = decode_latents(self.pipe, latents)
        final_res = self.auditor.audit_pil(final_pil, prompt, 0.0)
        ctrl_res  = self.auditor.audit_pil(first_pil or final_pil, prompt, 0.0)

        metrics = {
            "prompt":          prompt,
            "method":          method,
            "interventions":   interventions,
            "adv_final":       final_res["adv_prob"],
            "adv_ctrl":        ctrl_res["adv_prob"],
            "adv_improvement": ctrl_res["adv_prob"] - final_res["adv_prob"],
            "faithfulness":    final_res["faithfulness"],
            "seam_quality":    final_res["seam_quality"],
        }

        if cfg.get("save_images", False):
            os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)
            slug = prompt[:40].replace(" ", "_").replace("/", "")
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
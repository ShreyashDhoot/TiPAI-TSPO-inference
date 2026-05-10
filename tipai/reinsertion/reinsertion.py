"""
reinsertion/reinsertion.py
──────────────────────────
Four reinsertion strategies for injecting an inpainted winner back into the
live diffusion trajectory.

Only SD3_NULL_TEXT is recommended for production (per the spec). The others
are retained for ablation comparisons.

Public API
----------
REINSERTION_METHODS : dict[str, callable]
    Keys: 'SD0_DDPM', 'SD1_DDIM_FWD', 'SD2_DDIM_INV', 'SD3_NULL_TEXT'

reinsert(method, pipe, base_latents, winner_pil, mask_pil,
         t_norm, t_idx, text_emb, cfg, null_cache) → torch.Tensor
    Returns the updated latent tensor (same shape as base_latents).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Shared latent-space utilities
# ─────────────────────────────────────────────────────────────────────────────

def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        img = pipe.vae.decode(latents / 0.18215).sample
        img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((img[0] * 255).astype("uint8"))


def pil_to_latent(pipe, pil: Image.Image) -> torch.Tensor:
    device = next(pipe.vae.parameters()).device
    dtype  = next(pipe.vae.parameters()).dtype
    t = transforms.ToTensor()(pil.resize((512, 512)))
    t = (t * 2 - 1).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        return pipe.vae.encode(t).latent_dist.sample() * 0.18215


def make_mask_tensor(mask_pil: Image.Image, size: int = 64) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    return (
        transforms.ToTensor()(mask_pil.resize((size, size)))
        .to(device, dtype=dtype)
        .unsqueeze(0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SD0 — DDPM: add noise, blend, continue (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd0_ddpm(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    z_edit  = pil_to_latent(pipe, winner_pil)
    t       = pipe.scheduler.timesteps[t_idx]
    noise   = torch.randn_like(base_latents)
    z_noisy = pipe.scheduler.add_noise(z_edit, noise, t)
    mask_t  = make_mask_tensor(mask_pil)
    return (1 - mask_t) * base_latents + mask_t * z_noisy


# ─────────────────────────────────────────────────────────────────────────────
# SD1 — DDIM forward: encode + alpha-blend by t_norm
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd1_ddim_fwd(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    z_edit = pil_to_latent(pipe, winner_pil)
    mask_t = make_mask_tensor(mask_pil)
    alpha  = 1.0 - t_norm
    return (1 - alpha * mask_t) * base_latents + alpha * mask_t * z_edit


# ─────────────────────────────────────────────────────────────────────────────
# SD2 — DDIM inversion: manifold-aware short inversion before blend
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd2_ddim_inv(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    from diffusers import DDIMInverseScheduler

    z_edit = pil_to_latent(pipe, winner_pil)
    device = base_latents.device
    dtype  = base_latents.dtype
    d      = cfg.get("ddim_inv_steps", 10)

    inv_sch = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sch.set_timesteps(d, device=device)
    z_inv = z_edit.clone()
    with torch.no_grad():
        for t_inv in inv_sch.timesteps:
            li  = pipe.scheduler.scale_model_input(z_inv, t_inv)
            np_ = pipe.unet(li, t_inv, encoder_hidden_states=text_emb[-1:]).sample
            z_inv = inv_sch.step(np_, t_inv, z_inv).prev_sample

    mask_t = make_mask_tensor(mask_pil)
    alpha  = 1.0 - t_norm
    return (1 - alpha * mask_t) * base_latents + alpha * mask_t * z_inv


# ─────────────────────────────────────────────────────────────────────────────
# SD3 — Null-text inversion (PRODUCTION / recommended)
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd3_null_text(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache: dict | None = None,
):
    """
    Null-text inversion (Mokady et al. 2022) for mid-trajectory reinsertion.

    Optimises the unconditional embedding so DDIM inversion is near-lossless,
    then blends the aligned edit latent back via a mask.

    null_cache : pass the same dict across all audit steps so the optimised
                 null embedding is reused when revisiting the same t_idx.
    """
    if null_cache is None:
        null_cache = {}

    z_edit = pil_to_latent(pipe, winner_pil)
    device = base_latents.device
    dtype  = base_latents.dtype

    if t_idx not in null_cache:
        # ── optimise the null embedding ──────────────────────────────────────
        uncond_toks = pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            null_emb = pipe.text_encoder(
                uncond_toks.input_ids.to(device)
            )[0].clone().float()

        null_emb = nn.Parameter(null_emb)
        opt      = torch.optim.AdamW([null_emb], lr=cfg.get("null_lr", 0.01))
        t_val    = pipe.scheduler.timesteps[t_idx]

        for _ in range(cfg.get("null_opt_steps", 10)):
            opt.zero_grad()
            li  = pipe.scheduler.scale_model_input(z_edit.float(), t_val)
            np_ = pipe.unet(
                li.to(dtype), t_val,
                encoder_hidden_states=null_emb.to(dtype),
            ).sample
            pred = pipe.scheduler.step(
                np_.float(), t_val, z_edit.float()
            ).pred_original_sample
            F.mse_loss(pred, z_edit.float()).backward()
            opt.step()

        null_cache[t_idx] = null_emb.detach().to(dtype)

    # ── alpha-blend aligned edit into trajectory ──────────────────────────────
    mask_t = make_mask_tensor(mask_pil)
    alpha  = 1.0 - t_norm
    return (1 - alpha * mask_t) * base_latents + alpha * mask_t * z_edit


# ─────────────────────────────────────────────────────────────────────────────
# Registry + unified entry point
# ─────────────────────────────────────────────────────────────────────────────

REINSERTION_METHODS = {
    "SD0_DDPM":      reinsert_sd0_ddpm,
    "SD1_DDIM_FWD":  reinsert_sd1_ddim_fwd,
    "SD2_DDIM_INV":  reinsert_sd2_ddim_inv,
    "SD3_NULL_TEXT": reinsert_sd3_null_text,
}


def reinsert(
    method:       str,
    pipe,
    base_latents: torch.Tensor,
    winner_pil:   Image.Image,
    mask_pil:     Image.Image,
    t_norm:       float,
    t_idx:        int,
    text_emb:     torch.Tensor,
    cfg:          dict,
    null_cache:   dict | None = None,
) -> torch.Tensor:
    """
    Dispatch to the selected reinsertion method.

    Parameters
    ----------
    method       : one of REINSERTION_METHODS keys
    pipe         : base SD pipeline (provides scheduler, unet, vae)
    base_latents : current latent tensor in the diffusion trajectory
    winner_pil   : tournament-winning inpainted image (512×512 RGB PIL)
    mask_pil     : binary inpaint mask (512×512 L PIL)
    t_norm       : normalised timestep [0,1]
    t_idx        : integer index into pipe.scheduler.timesteps
    text_emb     : CFG text embedding pair [2, 77, 768]
    cfg          : config dict (keys: null_lr, null_opt_steps, ddim_inv_steps, …)
    null_cache   : persistent dict for SD3 null-embedding cache

    Returns
    -------
    torch.Tensor  updated latent (same shape/device/dtype as base_latents)
    """
    if method not in REINSERTION_METHODS:
        raise ValueError(
            f"Unknown reinsertion method '{method}'. "
            f"Choose from: {list(REINSERTION_METHODS)}"
        )
    fn = REINSERTION_METHODS[method]
    return fn(
        pipe, base_latents, winner_pil, mask_pil,
        t_norm, t_idx, text_emb, cfg, null_cache=null_cache,
    )

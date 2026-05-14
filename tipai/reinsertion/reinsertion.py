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

import warnings
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
    """
    Encode a PIL image to latents using pipe's VAE.
    Resizes to match the VAE's expected input based on base_latents spatial dims —
    caller is responsible for passing the correctly-sized PIL.
    The hardcoded 512 resize has been removed; size is preserved as-is.
    """
    device = next(pipe.vae.parameters()).device
    dtype  = next(pipe.vae.parameters()).dtype
    t = transforms.ToTensor()(pil)
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
# Family-aware UNet step helper
# ─────────────────────────────────────────────────────────────────────────────

def _unet_step(pipe, latents: torch.Tensor, t, text_emb, cfg_scale: float) -> torch.Tensor:
    """
    Single CFG-guided UNet denoising step, handles SD-1.x and SDXL.

    For SDXL (detected via presence of text_encoder_2), text_emb must be the
    dict produced by _encode_prompt_for_pipe in safe_diffusion.py, containing:
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds

    For SD-1.x, text_emb is the standard (2, 77, 768) tensor.
    """
    is_xl = hasattr(pipe, "text_encoder_2")

    if is_xl:
        pe   = text_emb["prompt_embeds"]
        npe  = text_emb["negative_prompt_embeds"]
        ppe  = text_emb["pooled_prompt_embeds"]
        npp  = text_emb["negative_pooled_prompt_embeds"]
        H    = latents.shape[2] * 8
        W    = latents.shape[3] * 8
        time_ids = torch.tensor(
            [[H, W, 0, 0, H, W]], device=latents.device, dtype=pe.dtype
        )
        add_cond = {"text_embeds": ppe, "time_ids": time_ids}
        add_unc  = {"text_embeds": npp, "time_ids": time_ids}
        li   = torch.cat([latents] * 2)
        li   = pipe.scheduler.scale_model_input(li, t)
        enc  = torch.cat([npe, pe])
        add  = {k: torch.cat([u, c]) for (k, u), (_, c) in zip(add_unc.items(), add_cond.items())}
        np_  = pipe.unet(li, t, encoder_hidden_states=enc, added_cond_kwargs=add).sample
    else:
        li   = torch.cat([latents] * 2)
        li   = pipe.scheduler.scale_model_input(li, t)
        np_  = pipe.unet(li, t, encoder_hidden_states=text_emb).sample

    u_n, c_n = np_.chunk(2)
    return u_n + cfg_scale * (c_n - u_n)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: resize winner + encode with base VAE, resize mask to latent size
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil):
    """
    1. Resize winner_pil to the pixel resolution implied by base_latents.
    2. Encode with base pipe VAE → z_edit in the correct latent space.
    3. Resize mask to latent spatial dims.
    Returns (z_edit, mask_t) both matching base_latents shape.
    """
    target_h = base_latents.shape[2] * 8
    target_w = base_latents.shape[3] * 8

    # Resize winner to base VAE input resolution
    if winner_pil.size != (target_w, target_h):
        winner_pil = winner_pil.resize((target_w, target_h), Image.LANCZOS)

    z_edit = pil_to_latent(pipe, winner_pil)

    # Guard: spatial mismatch after encode (should not fire after resize above)
    if z_edit.shape[2:] != base_latents.shape[2:]:
        warnings.warn(
            f"[reinsert] latent spatial mismatch {z_edit.shape} vs "
            f"{base_latents.shape} — check pil_to_latent"
        )
        z_edit = F.interpolate(
            z_edit.float(), size=base_latents.shape[2:],
            mode="bilinear", align_corners=False
        ).to(base_latents.dtype)

    # Guard: channel mismatch (SD3 16ch vs SD1x 4ch)
    if z_edit.shape[1] != base_latents.shape[1]:
        pad = torch.zeros_like(base_latents)
        pad[:, :z_edit.shape[1]] = z_edit
        z_edit = pad

    z_edit = z_edit.to(dtype=base_latents.dtype)

    # Mask → latent spatial dims
    mask_t = make_mask_tensor(mask_pil)
    if mask_t.shape[2:] != base_latents.shape[2:]:
        mask_t = F.interpolate(
            mask_t.float(), size=base_latents.shape[2:], mode="nearest"
        ).to(base_latents.dtype)

    return z_edit, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# SD0 — DDPM: encode winner with base VAE, add noise, blend
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd0_ddpm(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    z_edit, mask_t = _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil)
    noise   = torch.randn_like(base_latents)
    t       = pipe.scheduler.timesteps[t_idx]
    t_batch = t.reshape(1) if t.dim() == 0 else t
    z_noisy = pipe.scheduler.add_noise(z_edit, noise, t_batch)
    return (1 - mask_t) * base_latents + mask_t * z_noisy


# ─────────────────────────────────────────────────────────────────────────────
# SD1 — DDIM forward: encode + alpha-blend by t_norm
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd1_ddim_fwd(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    z_edit, mask_t = _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil)
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

    z_edit, mask_t = _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil)
    device = base_latents.device
    d      = cfg.get("ddim_inv_steps", 10)
    gs     = cfg.get("guidance_scale", 7.5)

    inv_sch = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_sch.set_timesteps(d, device=device)
    z_inv = z_edit.clone()
    with torch.no_grad():
        for t_inv in inv_sch.timesteps:
            guided = _unet_step(pipe, z_inv, t_inv, text_emb, gs)
            z_inv  = inv_sch.step(guided, t_inv, z_inv).prev_sample

    alpha = 1.0 - t_norm
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
    Works for both SD-1.x and SDXL via _unet_step.
    """
    if null_cache is None:
        null_cache = {}

    z_edit, mask_t = _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil)
    device = base_latents.device
    dtype  = base_latents.dtype
    gs     = cfg.get("guidance_scale", 7.5)

    if t_idx not in null_cache:
        is_xl = hasattr(pipe, "text_encoder_2")

        if is_xl:
            # For XL, optimise the pooled + sequence null embeddings together
            npe = text_emb["negative_prompt_embeds"].clone().float()
            npp = text_emb["negative_pooled_prompt_embeds"].clone().float()
            null_seq  = nn.Parameter(npe)
            null_pool = nn.Parameter(npp)
            opt = torch.optim.AdamW([null_seq, null_pool], lr=cfg.get("null_lr", 0.01))
        else:
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
            null_seq  = nn.Parameter(null_emb)
            null_pool = None
            opt = torch.optim.AdamW([null_seq], lr=cfg.get("null_lr", 0.01))

        t_val = pipe.scheduler.timesteps[t_idx]

        for _ in range(cfg.get("null_opt_steps", 10)):
            opt.zero_grad()

            if is_xl:
                # Build optimisable text_emb dict for _unet_step
                opt_text_emb = {
                    "prompt_embeds":              text_emb["prompt_embeds"],
                    "negative_prompt_embeds":     null_seq.to(dtype),
                    "pooled_prompt_embeds":       text_emb["pooled_prompt_embeds"],
                    "negative_pooled_prompt_embeds": null_pool.to(dtype),
                }
            else:
                # Stack null + cond for CFG
                cond = text_emb[-1:] if text_emb.dim() == 3 else text_emb
                opt_text_emb = torch.cat([null_seq.to(dtype), cond], dim=0)

            guided = _unet_step(pipe, z_edit.float(), t_val, opt_text_emb, gs)
            pred   = pipe.scheduler.step(
                guided.float(), t_val, z_edit.float()
            ).pred_original_sample
            F.mse_loss(pred, z_edit.float()).backward()
            opt.step()

        if is_xl:
            null_cache[t_idx] = {
                "seq":  null_seq.detach().to(dtype),
                "pool": null_pool.detach().to(dtype),
            }
        else:
            null_cache[t_idx] = null_seq.detach().to(dtype)

    alpha = 1.0 - t_norm
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
    text_emb,
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
    winner_pil   : tournament-winning inpainted image (any size — auto-resized)
    mask_pil     : binary inpaint mask (any size — auto-resized to latent dims)
    t_norm       : normalised timestep [0,1]
    t_idx        : integer index into pipe.scheduler.timesteps
    text_emb     : SD-1.x: (2, 77, 768) tensor
                   SDXL:   dict with prompt_embeds / negative_prompt_embeds /
                           pooled_prompt_embeds / negative_pooled_prompt_embeds
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

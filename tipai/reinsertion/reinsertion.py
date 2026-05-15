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
from PIL import Image, ImageFilter
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Shared latent-space utilities
# ─────────────────────────────────────────────────────────────────────────────

def decode_latents(pipe, latents: torch.Tensor) -> Image.Image:
    """SD-1.x decode only. For SD3x / SDXL use _decode_pil in safe_diffusion.py."""
    with torch.no_grad():
        img = pipe.vae.decode(latents / 0.18215).sample
        img = (img / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((img[0] * 255).astype("uint8"))


def pil_to_latent(pipe, pil: Image.Image) -> torch.Tensor:
    """
    Encode a PIL image to latents using pipe's VAE.

    Handles both SD-1.x (scaling_factor=0.18215, no shift) and SD3.x
    (scaling_factor≈1.5305, shift_factor≈0.0609).  The factors are read
    from vae.config so this works for any checkpoint without hardcoding.

    SD3 encode formula (mirrors diffusers StableDiffusion3Pipeline._encode_vae_image):
        latents = (vae.encode(x).latent_dist.sample() - shift_factor) * scaling_factor
    """
    device = next(pipe.vae.parameters()).device
    dtype  = next(pipe.vae.parameters()).dtype
    t = transforms.ToTensor()(pil)
    t = (t * 2 - 1).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(t).latent_dist.sample()

    scaling = pipe.vae.config.scaling_factor
    shift   = getattr(pipe.vae.config, "shift_factor", None) or 0.0
    # Apply shift before scaling — matches diffusers SD3 encode convention
    return (latents - shift) * scaling


def make_mask_tensor(
    mask_pil: Image.Image,
    target_h: int,
    target_w: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Resize mask_pil to (target_h, target_w) and return as a float tensor
    with shape (1, 1, target_h, target_w).

    target_h / target_w should be the latent spatial dims (e.g. 128×128 for
    a 1024×1024 SD3 image, 64×64 for a 512×512 SD1x image).
    """
    return (
        transforms.ToTensor()(mask_pil.resize((target_w, target_h), Image.NEAREST))
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

def _prepare_edit_latents(pipe, base_latents, winner_pil, mask_pil, sharpen: bool = False):
    """
    1. Optionally sharpen winner_pil (helps recover detail lost in 512→1024 upscale).
    2. Resize winner_pil to the pixel resolution implied by base_latents.
    3. Encode with base pipe VAE → z_edit in the correct latent space.
    4. Resize mask to latent spatial dims.
    Returns (z_edit, mask_t) both matching base_latents shape/device/dtype.

    Notes
    -----
    pil_to_latent now reads vae.config.scaling_factor and shift_factor, so
    this works correctly for both SD1x (4ch) and SD3x (16ch) VAEs.
    The channel-mismatch zero-pad branch below is a last-resort guard only —
    it should never fire if pil_to_latent is paired with the correct VAE.
    """
    target_h = base_latents.shape[2] * 8
    target_w = base_latents.shape[3] * 8

    if winner_pil.size != (target_w, target_h):
        winner_pil = winner_pil.resize((target_w, target_h), Image.LANCZOS)

    # Optional sharpening: compensates for softness introduced by the
    # 512→1024 LANCZOS upscale of the SD1.x inpainter output.
    if sharpen:
        winner_pil = winner_pil.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2)
        )

    z_edit = pil_to_latent(pipe, winner_pil)

    # Guard: spatial mismatch after encode (should not fire after resize above)
    if z_edit.shape[2:] != base_latents.shape[2:]:
        warnings.warn(
            f"[reinsert] latent spatial mismatch {z_edit.shape} vs "
            f"{base_latents.shape} — interpolating"
        )
        z_edit = F.interpolate(
            z_edit.float(), size=base_latents.shape[2:],
            mode="bilinear", align_corners=False
        ).to(base_latents.dtype)

    # Guard: channel mismatch — should not fire if using correct VAE
    if z_edit.shape[1] != base_latents.shape[1]:
        warnings.warn(
            f"[reinsert] channel mismatch: z_edit={z_edit.shape[1]}ch "
            f"vs base_latents={base_latents.shape[1]}ch — check that "
            f"pil_to_latent is using the right VAE for this model family"
        )
        pad = torch.zeros_like(base_latents)
        pad[:, :z_edit.shape[1]] = z_edit
        z_edit = pad

    z_edit = z_edit.to(device=base_latents.device, dtype=base_latents.dtype)

    # Mask → latent spatial dims, matching base_latents device/dtype
    lat_h, lat_w = base_latents.shape[2], base_latents.shape[3]
    mask_t = make_mask_tensor(
        mask_pil, lat_h, lat_w,
        device=base_latents.device,
        dtype=base_latents.dtype,
    )

    return z_edit, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Shared blend helper (used by SD4 and others)
# ─────────────────────────────────────────────────────────────────────────────

def _blend(base_latents, z_inv, mask_t, alpha, hard_threshold: float = 0.0):
    """
    Blend z_inv into base_latents inside the mask region.

    When alpha < hard_threshold, does a hard swap (no interpolation):
        result = base_latents outside mask, z_inv inside mask.
    Otherwise does a soft blend:
        result = (1-alpha)*base_latents + alpha*(base_latents*(1-mask) + z_inv*mask)

    The z_inv_masked construction ensures the outside-mask region of z_inv
    never leaks into base_latents regardless of alpha.

    Parameters
    ----------
    base_latents    : noisy latent from the diffusion trajectory
    z_inv           : flow-projected (or clean) winner latent
    mask_t          : float tensor in [0,1], shape (1,1,H,W)
    alpha           : blend weight; for SD3x this should be t_norm (NOT 1-t_norm)
    hard_threshold  : alpha values below this use a hard swap instead of blending
    """
    # Hard swap: maximum sharpness, use at late steps when alpha is small
    if alpha < hard_threshold:
        return base_latents * (1.0 - mask_t) + z_inv * mask_t

    # Soft blend: z_inv only contributes inside the mask
    z_inv_masked = base_latents * (1.0 - mask_t) + z_inv * mask_t
    return (1.0 - alpha) * base_latents + alpha * z_inv_masked


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
    # Integrate in float32 for stability, but feed the transformer its native dtype.
    z_inv = z_edit.float().clone()
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
                    "prompt_embeds":              text_emb["prompt_embeds"].detach(),
                    "negative_prompt_embeds":     null_seq.to(dtype),
                    "pooled_prompt_embeds":       text_emb["pooled_prompt_embeds"].detach(),
                    "negative_pooled_prompt_embeds": null_pool.to(dtype),
                }
            else:
                # Stack null + cond for CFG
                cond = text_emb[-1:] if text_emb.dim() == 3 else text_emb
                opt_text_emb = torch.cat([null_seq.to(dtype), cond], dim=0)

            guided = _unet_step(pipe, z_edit.to(dtype), t_val, opt_text_emb, gs)
            pred   = pipe.scheduler.step(
                guided.to(dtype), t_val, z_edit.to(dtype)
            ).pred_original_sample
            F.mse_loss(pred, z_edit.to(dtype)).backward()
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
# SD4 — Flow inversion: forward ODE projection for SD3.5 (rectified flow)
# ─────────────────────────────────────────────────────────────────────────────

def reinsert_sd3_flow_inv(
    pipe, base_latents, winner_pil, mask_pil,
    t_norm, t_idx, text_emb, cfg, null_cache=None,
):
    """
    Flow-matching reinsertion for SD3.5 (rectified flow / FlowMatchEulerDiscrete).

    Encodes the winner PIL to a clean latent z_0, then optionally runs the
    flow ODE forward from t=0 to t_current using the transformer's velocity
    field to project z_0 onto the noisy manifold at the correct timestep.

    Rectified flow ODE (forward, data→noise):
        x_t = (1-t)·x_0 + t·ε
        dx/dt = v_θ(x_t, t)
    We integrate forward with Euler steps from t=0 to t_current.

    Blend behaviour
    ───────────────
    For SD3.5, t_norm IS the sigma (FlowMatch timesteps run 1→0), so:
        alpha = t_norm    (high early = noisy, low late = clean)

    When alpha < flow_inv_hard_threshold: hard swap (maximum sharpness).
    When flow_inv_steps == 0:             skip ODE, blend clean z_edit directly.
    When t_norm < flow_inv_clean_threshold and steps > 0: also skip ODE
        (we're so late that adding noise then removing it degrades quality).

    Config keys (all optional)
    ──────────────────────────
    flow_inv_steps           : int   — Euler steps for ODE (0 = skip, default 10)
    flow_inv_hard_threshold  : float — alpha below this → hard swap (default 0.25)
    flow_inv_clean_threshold : float — t_norm below this → skip ODE (default 0.3)
    flow_inv_sharpen         : bool  — sharpen winner PIL before encoding (default True)
    flow_inv_alpha           : float — override alpha (default None = use t_norm)
    """
    sharpen = cfg.get("flow_inv_sharpen", True)
    z_edit, mask_t = _prepare_edit_latents(
        pipe, base_latents, winner_pil, mask_pil, sharpen=sharpen
    )

    print(
        f"  [FlowInv] z_edit=[{z_edit.min():.3f}, {z_edit.max():.3f}]"
        f"  base=[{base_latents.min():.3f}, {base_latents.max():.3f}]"
        f"  t_norm={t_norm:.3f}  t_current={pipe.scheduler.timesteps[t_idx].item():.3f}"
    )

    device = base_latents.device
    dtype  = base_latents.dtype

    # ── Alpha: for SD3x flow, t_norm IS sigma (1=noisy, 0=clean) ─────────────
    alpha = cfg.get("flow_inv_alpha", None)
    if alpha is None:
        alpha = t_norm   # NOT (1 - t_norm) — that was the original direction bug

    hard_threshold  = cfg.get("flow_inv_hard_threshold",  0.25)
    clean_threshold = cfg.get("flow_inv_clean_threshold", 0.30)
    inv_steps       = cfg.get("flow_inv_steps", 10)

    # ── Fast path: skip ODE entirely ─────────────────────────────────────────
    # Conditions: explicitly disabled (inv_steps=0), or late enough in
    # denoising that adding noise via ODE then relying on remaining steps to
    # clean it would only hurt sharpness.
    skip_ode = (inv_steps == 0) or (t_norm < clean_threshold)

    if skip_ode:
        print(f"  [FlowInv] Skipping ODE (inv_steps={inv_steps}, t_norm={t_norm:.3f} < clean_threshold={clean_threshold})")
        return _blend(base_latents, z_edit, mask_t, alpha, hard_threshold)

    # ── Flow ODE forward integration ──────────────────────────────────────────
    # Sub-schedule: linearly from 0 (clean) to t_current (current noise level).
    t_current  = pipe.scheduler.timesteps[t_idx].item()  # sigma in (0, 1]
    sub_sigmas = torch.linspace(0.0, t_current, inv_steps + 1, device=device, dtype=dtype)

    pe,  npe = text_emb["prompt_embeds"],  text_emb["negative_prompt_embeds"]
    ppe, npp = text_emb["pooled_prompt_embeds"], text_emb["negative_pooled_prompt_embeds"]
    gs = cfg.get("guidance_scale", 4.5)

    z_inv = z_edit.clone()

    with torch.no_grad():
        for step_idx in range(inv_steps):
            t_s = sub_sigmas[step_idx]
            t_e = sub_sigmas[step_idx + 1]
            dt  = t_e - t_s

            # SD3.5 MMDiT was trained with timesteps in [0, 1000].
            # FlowMatchEulerDiscrete sigmas are in [0, 1] — scale up.
            t_batch = (t_s * 1000.0).expand(z_inv.shape[0]).to(device, dtype=z_inv.dtype)

            tr_dtype = next(pipe.transformer.parameters()).dtype
            li   = torch.cat([z_inv.to(dtype=tr_dtype)] * 2)
            enc  = torch.cat([npe, pe])
            pool = torch.cat([npp, ppe])
            t_in = t_batch.repeat(2)

            vel = pipe.transformer(
                li,
                timestep=t_in,
                encoder_hidden_states=enc,
                pooled_projections=pool,
            ).sample

            u_vel, c_vel = vel.chunk(2)
            guided_vel   = (u_vel + gs * (c_vel - u_vel)).float()

            # Euler step forward along the flow (clean → noisy direction)
            z_inv = z_inv + guided_vel * dt

            print(
                f"  [FlowInv ODE] step={step_idx}  t_s={t_s.item():.4f}"
                f"  vel_norm={guided_vel.norm().item():.3f}"
                f"  z_inv_norm={z_inv.norm().item():.3f}"
            )

    z_inv = z_inv.to(device=base_latents.device, dtype=base_latents.dtype)

    return _blend(base_latents, z_inv, mask_t, alpha, hard_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Registry + unified entry point
# ─────────────────────────────────────────────────────────────────────────────

REINSERTION_METHODS = {
    "SD0_DDPM":      reinsert_sd0_ddpm,
    "SD1_DDIM_FWD":  reinsert_sd1_ddim_fwd,
    "SD2_DDIM_INV":  reinsert_sd2_ddim_inv,
    "SD3_NULL_TEXT": reinsert_sd3_null_text,
    "SD4_FLOW_INV":  reinsert_sd3_flow_inv,   # correct method for SD3.5 / rectified flow
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
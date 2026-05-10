"""
utils/diffusion_utils.py
─────────────────────────
Shared diffusion-space utilities used across modules.

Functions
---------
encode_prompt(pipe, prompt, device)      → torch.Tensor  [2, 77, 768]
build_mask(hmap, feather_sigma, pct)     → PIL.Image  (L mode 512×512)
noise_aware_heatmap(hmap, t_val, sched) → np.ndarray
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


def encode_prompt(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    """
    Encode a text prompt into a CFG embedding pair [uncond; cond].

    Returns
    -------
    torch.Tensor  shape [2, max_len, 768]
    """
    max_len = pipe.tokenizer.model_max_length

    def _enc(text: str) -> torch.Tensor:
        toks = pipe.tokenizer(
            [text],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return pipe.text_encoder(toks.input_ids.to(device))[0]   # [1, 77, 768]

    return torch.cat([_enc(""), _enc(prompt)])   # [2, 77, 768]


def build_mask(
    hmap:          np.ndarray,
    feather_sigma: float = 5.0,
    pct:           float = 65.0,
) -> Image.Image:
    """
    Convert a raw float32 heatmap (any resolution) into a feathered binary
    inpaint mask (L-mode PIL, 512×512).

    Parameters
    ----------
    hmap          : float32 array, shape (H, W)
    feather_sigma : Gaussian blur radius for soft edge
    pct           : percentile threshold — top-(100-pct)% pixels are masked

    Returns
    -------
    PIL.Image  'L' mode, 0 = keep, 255 = inpaint
    """
    h512 = cv2.resize(hmap, (512, 512))
    bm   = (h512 >= np.percentile(h512, pct)).astype(np.float32)
    k    = int(feather_sigma * 4) | 1   # must be odd
    soft = cv2.GaussianBlur(bm, (k, k), feather_sigma)
    arr  = (soft > 0.5).astype(np.float32)
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("L")


def noise_aware_heatmap(
    hmap:      np.ndarray,
    t_val:     int | torch.Tensor,
    scheduler,
) -> np.ndarray:
    """
    Scale the adversarial heatmap by (1 - alpha_t^0.5) so earlier timesteps
    (higher noise, lower SNR) produce smaller masks.

    Parameters
    ----------
    hmap      : raw heatmap from auditor
    t_val     : integer timestep or scalar tensor
    scheduler : pipe.scheduler  (must have alphas_cumprod)

    Returns
    -------
    np.ndarray  scaled heatmap same shape as input
    """
    t_int   = int(t_val.item()) if hasattr(t_val, "item") else int(t_val)
    snr_amp = float(scheduler.alphas_cumprod[t_int].sqrt())
    return hmap * (1.0 - snr_amp)

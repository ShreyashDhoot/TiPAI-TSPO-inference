"""
inpainting/inpainter.py
───────────────────────
Wrapper around any SD-inpaint pipeline.
Supports:
  - swapping the base inpainter model (any HF id or local path)
  - loading / hot-swapping soft-tensor LoRA weights
  - prompt sanitisation by harm class

Public API
----------
build_inpainter(model_id, lora_path, lora_scale, device, dtype)
    → StableDiffusionInpaintPipeline

run_inpainting(inpaint_pipe, base_pil, mask_pil, prompt, harm_class, knob)
    → PIL.Image
"""

from __future__ import annotations
import re

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from policy.tspo_policy import KnobSet

# ── Toxic lexicon for prompt sanitisation ─────────────────────────────────────
_TOXIC = {
    "nudity": [
        r"naked", r"nude", r"nsfw", r"sex", r"boobs?", r"penis",
        r"slut", r"fucked", r"horny", r"breasts?", r"genitalia", r"intercourse",
    ],
    "violence": [
        r"blood[y]?", r"gore", r"gun", r"ak47", r"shoot(ing)?",
        r"dead", r"kill(ing)?", r"murder", r"fight(ing)?", r"mafia", r"weapons?",
    ],
}

_NEGATIVE = {
    "nudity":   "naked, nude, nsfw, explicit, revealing, bare skin, breasts, genitalia, cleavage",
    "violence": "blood, gore, weapons, gun, knife, fighting, violence, injury, war, dead, shooting",
    "safe":     "blurry, watermark, text",
}
_NEGATIVE_SUFFIX = ", blurry, watermark, text, worst quality, deformed, bad anatomy"


def _sanitize_prompt(prompt: str, harm_class: str) -> str:
    clean = prompt.lower()
    for pattern in _TOXIC.get(harm_class, []):
        clean = re.sub(rf"\b{pattern}\b", "", clean)
    clean = " ".join(clean.split())
    return clean if len(clean) >= 3 else "a safe, normal scene"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_inpainter(
    model_id:    str          = "runwayml/stable-diffusion-inpainting",
    lora_path:   str | None   = None,
    lora_scale:  float        = 0.8,
    vae_from                  = None,   # pass pipe.vae to share VAE with base pipe
    device:      torch.device | str = "cuda",
    dtype:       torch.dtype         = torch.float16,
) -> StableDiffusionInpaintPipeline:
    """
    Build (or rebuild) the inpaint pipeline.

    Parameters
    ----------
    model_id   : HuggingFace repo id or local path for the inpaint model.
    lora_path  : optional path to LoRA weights (.safetensors or .bin).
                 Pass None to skip LoRA loading.
    lora_scale : LoRA merge scale (default 0.8).
    vae_from   : if provided, replaces the inpainter's VAE with this object
                 (keeps shared latent space with the base pipe).
    device     : target device.
    dtype      : torch dtype (fp16 on GPU, fp32 on CPU).

    Returns
    -------
    StableDiffusionInpaintPipeline  ready for inference.
    """
    device = torch.device(device) if isinstance(device, str) else device

    pip = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    pip.safety_checker = None

    if vae_from is not None:
        pip.vae = vae_from

    if lora_path is not None:
        print(f"[Inpainter] Loading LoRA: {lora_path}  scale={lora_scale}")
        pip.load_lora_weights(lora_path)
        pip.fuse_lora(lora_scale=lora_scale)

    return pip


def swap_lora(
    inpaint_pipe: StableDiffusionInpaintPipeline,
    new_lora_path: str,
    new_lora_scale: float = 0.8,
) -> StableDiffusionInpaintPipeline:
    """
    Unfuse current LoRA (if any) and load a new one.
    Returns the same pipeline object (mutated in-place).
    """
    try:
        inpaint_pipe.unfuse_lora()
    except Exception:
        pass   # no LoRA was fused — fine
    inpaint_pipe.load_lora_weights(new_lora_path)
    inpaint_pipe.fuse_lora(lora_scale=new_lora_scale)
    print(f"[Inpainter] Swapped LoRA → {new_lora_path}  scale={new_lora_scale}")
    return inpaint_pipe


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inpainting(
    inpaint_pipe:  StableDiffusionInpaintPipeline,
    base_pil:      Image.Image,
    mask_pil:      Image.Image,
    prompt:        str,
    harm_class:    str,
    knob:          KnobSet,
    n_steps:       int   = 20,
    prompt_mode:   str   = "safe",   # "safe" | "original"
    device:        torch.device | str = "cuda",
) -> Image.Image:
    """
    Run a single inpainting pass with the given knob settings.

    Parameters
    ----------
    inpaint_pipe : the loaded inpaint pipeline
    base_pil     : 512×512 source image
    mask_pil     : 512×512 L-mode binary mask (white = inpaint region)
    prompt       : original generation prompt
    harm_class   : 'nudity' | 'violence' | 'safe'
    knob         : KnobSet produced by get_knobs()
    n_steps      : number of inpainting diffusion steps
    prompt_mode  : "safe" scrubs toxic words; "original" keeps the prompt as-is

    Returns
    -------
    PIL.Image  inpainted 512×512 RGB image
    """
    p   = _sanitize_prompt(prompt, harm_class) if prompt_mode == "safe" else prompt
    neg = _NEGATIVE.get(harm_class, "blurry, watermark, nsfw") + _NEGATIVE_SUFFIX
    gen = torch.Generator(device=device).manual_seed(knob.seed_offset)

    # inversion_depth is an int in [1, 10] (KNOB_BOUNDS).
    # SD inpaint `strength` must be a float in [0.0, 1.0].
    # Map linearly: depth 1 → 0.3 (light repaint), depth 10 → 1.0 (full repaint).
    strength = 0.3 + (knob.inversion_depth - 1) / 9.0 * 0.7   # [0.3, 1.0]
    strength = float(max(0.0, min(1.0, strength)))              # hard clamp

    return inpaint_pipe(
        prompt              = p,
        negative_prompt     = neg,
        image               = base_pil,
        mask_image          = mask_pil,
        guidance_scale      = max(knob.cfg_scale, 8.5),
        strength            = strength,
        num_inference_steps = n_steps,
        generator           = gen,
    ).images[0]
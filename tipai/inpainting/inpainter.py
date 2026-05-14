"""
inpainting/inpainter.py
───────────────────────
Wrapper around any SD-inpaint pipeline.
Supports:
  - swapping the base inpainter model (any HF id or local path)
  - loading / hot-swapping soft-tensor LoRA weights
  - hard-coded negative prompt only (no positive prompt sent to inpainter)

Design decision
---------------
The inpainter receives NO positive prompt — only a hard-coded negative prompt
that lists harm concepts for the detected class.  This removes the risk of the
orthogonal or sanitised positive prompt leaking adversarial concept directions
into the inpainted region.  Empirically, inpainting models fill masked regions
coherently from surrounding context even without a positive prompt; negative
guidance is sufficient to steer away from the unsafe concept.

The old `_build_prompt_embeds` orthogonal-embedding helper has been removed.

Public API
----------
build_inpainter(model_id, lora_path, lora_scale, device, dtype)
    → StableDiffusionInpaintPipeline

run_inpainting(inpaint_pipe, base_pil, mask_pil, harm_class, knob, n_steps, device)
    → PIL.Image
"""

from __future__ import annotations

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

from policy.tspo_policy import KnobSet

# ── Hard-coded negative prompts by harm class ─────────────────────────────────
# These are the ONLY prompts passed to the inpainter.
# No positive prompt is ever supplied.
_NEGATIVE_PROMPT: dict[str, str] = {
    "nudity": (
        "naked, nude, nsfw, explicit, revealing, bare skin, breasts, genitalia, "
        "cleavage, underwear, lingerie, pornographic,deformed , bad anatomy"
    ),
    "violence": (
        "blood, gore, weapons, gun, knife, fighting, violence, injury, war, "
        "dead, shooting, bullet, murder, wound, deformed, bad anatomy"
    ),
    "safe": (
        "blurry, watermark, text, worst quality, deformed, bad anatomy"
    ),
}
_NEGATIVE_SUFFIX = ", blurry, watermark, text, worst quality, deformed, bad anatomy"


def _negative_for(harm_class: str) -> str:
    base = _NEGATIVE_PROMPT.get(harm_class, _NEGATIVE_PROMPT["safe"])
    if harm_class in ("nudity", "violence"):
        return base + _NEGATIVE_SUFFIX
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def build_inpainter(
    model_id:    str          = "runwayml/stable-diffusion-inpainting",
    lora_path:   str | None   = None,
    lora_scale:  float        = 0.8,
    vae_from                  = None,   # explicitly NOT shared — see note below
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
    vae_from   : DEPRECATED — pass None.  The inpainter always uses its own VAE.
                 Sharing a VAE between the base generator and inpainter is
                 incorrect when they use different model families (e.g. SD-1.5 base
                 with XL inpainter) because their latent statistics differ.
    device     : target device.
    dtype      : torch dtype (fp16 on GPU, fp32 on CPU).

    Returns
    -------
    StableDiffusionInpaintPipeline  ready for inference.
    """
    if vae_from is not None:
        import warnings
        warnings.warn(
            "[Inpainter] vae_from is deprecated and ignored. "
            "The inpainter always uses its own VAE to avoid latent-space mismatches "
            "when base and inpainter are from different model families.",
            DeprecationWarning,
            stacklevel=2,
        )

    device = torch.device(device) if isinstance(device, str) else device

    pip = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    pip.safety_checker = None

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
# Inference  — NO positive prompt, hard-coded negative only
# ─────────────────────────────────────────────────────────────────────────────

def run_inpainting(
    inpaint_pipe:  StableDiffusionInpaintPipeline,
    base_pil:      Image.Image,
    mask_pil:      Image.Image,
    harm_class:    str,
    knob:          KnobSet,
    n_steps:       int   = 20,
    device:        torch.device | str = "cuda",
    # Legacy arguments silently ignored so call-sites need not change
    prompt:        str   = "",          # IGNORED — not passed to inpainter
    adv_prob:      float = 0.5,         # IGNORED — orthogonal path removed
    prompt_mode:   str   = "none",      # IGNORED
    orth_alpha:    float = 0.7,         # IGNORED
) -> Image.Image:
    """
    Run a single inpainting pass with the given knob settings.

    The inpainter receives NO positive prompt.  Only a hard-coded negative
    prompt (indexed by harm_class) is supplied so the model fills the masked
    region from visual context while being repelled from unsafe concepts.

    Parameters
    ----------
    inpaint_pipe : the loaded inpaint pipeline
    base_pil     : source image (any size — resized to 512×512 internally if needed)
    mask_pil     : L-mode binary mask (white = inpaint region; any size, auto-resized)
    harm_class   : 'nudity' | 'violence' | 'safe'
    knob         : KnobSet produced by get_knobs()
    n_steps      : number of inpainting diffusion steps
    device       : torch device

    Silently ignored (kept for call-site compat)
    -----------------------------------------------
    prompt, adv_prob, prompt_mode, orth_alpha

    Returns
    -------
    PIL.Image  inpainted RGB image at the same resolution as base_pil
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Ensure consistent 512×512 input for SD-1.x inpainters
    base_512 = base_pil.resize((512, 512), Image.LANCZOS)
    mask_512 = mask_pil.resize((512, 512), Image.NEAREST)

    neg = _negative_for(harm_class)
    gen = torch.Generator(device=device).manual_seed(knob.seed_offset)
    strength = 0.3 + (knob.inversion_depth - 1) / 9.0 * 0.7   # [0.3, 1.0]
    strength = float(max(0.0, min(1.0, strength)))

    print(
        f"  [Inpainter] harm_class={harm_class}  strength={strength:.3f}  "
        f"cfg={knob.cfg_scale:.1f}  n_steps={n_steps}  seed={knob.seed_offset}  "
        f"[NO positive prompt]"
    )

    result = inpaint_pipe(
        prompt              = "",          # empty — no positive prompt
        negative_prompt     = neg,
        image               = base_512,
        mask_image          = mask_512,
        guidance_scale      = max(knob.cfg_scale, 8.5),
        strength            = strength,
        num_inference_steps = n_steps,
        generator           = gen,
    ).images[0]

    # Resize back to original base_pil resolution if it was not 512×512
    if base_pil.size != (512, 512):
        result = result.resize(base_pil.size, Image.LANCZOS)

    return result

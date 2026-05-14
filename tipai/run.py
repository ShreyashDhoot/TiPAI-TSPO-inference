#!/usr/bin/env python3
"""
run.py
──────
CLI entry-point for TiPAI-TSPO safe image generation.

Usage
-----
    python run.py --prompt "a woman at the beach" --config config.yaml

    # Override seed at runtime
    python run.py --prompt "..." --seed 123

    # Swap inpainter LoRA without editing config.yaml
    python run.py --prompt "..." --lora weights/my_lora.safetensors

    # Use a different base SD model
    python run.py --prompt "..." --base-model "runwayml/stable-diffusion-v1-5"
"""

import argparse
import sys

from utils.config_loader import load_config
from pipeline.safe_diffusion import SafeDiffusionPipeline
from utils.hf_auth import resolve_hf_token


def parse_args():
    p = argparse.ArgumentParser(description="TiPAI-TSPO: safe image generation")
    p.add_argument("--prompt",     required=True,             help="Text prompt")
    p.add_argument("--config",     default="config.yaml",     help="Path to config.yaml")
    p.add_argument("--seed",       type=int, default=None,    help="RNG seed (overrides config)")
    p.add_argument("--base-model", default=None,              help="HF id or local path for base SD")
    p.add_argument("--lora",       default=None,              help="Path to inpainter LoRA weights")
    p.add_argument("--lora-scale", type=float, default=0.8,   help="LoRA merge scale")
    p.add_argument("--out",        default=None,              help="Output image path (overrides config)")
    p.add_argument("--hf-token",  default=None,              help="HuggingFace token for gated models (overrides HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # CLI overrides
    if args.base_model:
        cfg["base_sd_model"] = args.base_model
    if args.lora:
        cfg["inpainter_lora_path"]  = args.lora
        cfg["inpainter_lora_scale"] = args.lora_scale

    # Build pipeline (token resolved from --hf-token or HF_TOKEN env)
    sdp = SafeDiffusionPipeline(cfg, hf_token=args.hf_token)

    # Generate
    result = sdp.generate(args.prompt, seed=args.seed)

    # Save
    out_path = args.out or result.metrics.get("saved_to")
    if out_path:
        result.image.save(out_path)
        print(f"[Done] Saved → {out_path}")
    else:
        result.image.show()

    # Summary
    m = result.metrics
    print(
        f"\n── Summary ──────────────────────────────────\n"
        f"  Prompt        : {args.prompt}\n"
        f"  Method        : {m['method']}\n"
        f"  Interventions : {m['interventions']}\n"
        f"  Δ adv score   : {m['adv_improvement']:+.4f}\n"
        f"  Final adv_prob: {m['adv_final']:.4f}  "
        f"({'SAFE ✓' if result.final_safe else 'UNSAFE ✗'})\n"
        f"  Faithfulness  : {m['faithfulness']:.4f}\n"
        f"─────────────────────────────────────────────"
    )

    sys.exit(0 if result.final_safe else 1)


if __name__ == "__main__":
    main()

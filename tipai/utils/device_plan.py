"""
utils/device_plan.py
────────────────────
Device assignment strategy for multi-GPU servers.

SD-1.x and SDXL fit comfortably on a single A6000 (48 GB).
SD-3.5 Large Turbo / Medium does NOT — the MMDiT transformer alone is ~18 GB
in fp16, and with text encoders (T5-XXL ≈ 10 GB) the total is ~30+ GB.
With a 7× A6000 server we shard SD3x across two GPUs:
  - SD3x transformer + schedulers  → cuda:0   (largest component)
  - SD3x VAE + text encoders       → cuda:1   (moderate)
  - Auditor + Inpainter + Policy   → cuda:2   (fixed, never changes)

SD-1.x and SDXL run entirely on cuda:2 alongside the support models, since
they fit in the remaining headroom of a single A6000 after auditor+inpainter.

If fewer than 3 GPUs are available everything falls back to cuda:0.

Public API
----------
DevicePlan(family, n_gpus) → plan
    plan.base_device      torch.device  primary device for base generator
    plan.support_device   torch.device  auditor / inpainter / policy device
    plan.sd3_vae_device   torch.device  SD3 VAE + text encoders (may == base_device)
    plan.use_model_cpu_offload  bool    True when memory is very tight
    plan.summary()        str           human-readable assignment table
"""

from __future__ import annotations
import torch
from dataclasses import dataclass


@dataclass
class DevicePlan:
    base_device:           torch.device   # transformer/unet lives here
    support_device:        torch.device   # auditor, inpainter, policy
    sd3_vae_device:        torch.device   # SD3 VAE + text encoders
    use_model_cpu_offload: bool           # enable diffusers cpu offload hook

    def summary(self) -> str:
        lines = [
            "  ┌─ Device Assignment ──────────────────────────────────",
            f"  │  Base generator (transformer/UNet) → {self.base_device}",
            f"  │  SD3 VAE + text encoders           → {self.sd3_vae_device}",
            f"  │  Auditor / Inpainter / Policy      → {self.support_device}",
            f"  │  CPU offload                        → {self.use_model_cpu_offload}",
            "  └─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


def make_device_plan(family: str, n_gpus: int | None = None) -> DevicePlan:
    """
    Build a DevicePlan for the given model family.

    Parameters
    ----------
    family : 'sd1x' | 'sdxl' | 'sd3x'
    n_gpus : number of available CUDA devices (auto-detected if None)
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count()

    no_cuda = n_gpus == 0

    def dev(idx: int) -> torch.device:
        if no_cuda:
            return torch.device("cpu")
        return torch.device(f"cuda:{min(idx, n_gpus - 1)}")

    if family in ("sd1x", "sdxl"):
        # Fits on a single GPU. Put everything on the same device.
        # If we have ≥3 GPUs, use cuda:2 so SD3x can own cuda:0 and cuda:1
        # when the user later hot-swaps; otherwise just use cuda:0.
        d = dev(2) if n_gpus >= 3 else dev(0)
        return DevicePlan(
            base_device           = d,
            support_device        = d,
            sd3_vae_device        = d,
            use_model_cpu_offload = False,
        )

    if family == "sd3x":
        if n_gpus >= 3:
            # Ideal: transformer on cuda:0, VAE+encoders on cuda:1,
            # auditor/inpainter/policy on cuda:2
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(2),
                sd3_vae_device        = dev(1),
                use_model_cpu_offload = False,
            )
        elif n_gpus == 2:
            # Transformer on cuda:0, everything else on cuda:1
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(1),
                sd3_vae_device        = dev(1),
                use_model_cpu_offload = False,
            )
        elif n_gpus == 1:
            # Single GPU — enable cpu offload to avoid OOM
            return DevicePlan(
                base_device           = dev(0),
                support_device        = dev(0),
                sd3_vae_device        = dev(0),
                use_model_cpu_offload = True,
            )
        else:
            return DevicePlan(
                base_device           = torch.device("cpu"),
                support_device        = torch.device("cpu"),
                sd3_vae_device        = torch.device("cpu"),
                use_model_cpu_offload = False,
            )

    raise ValueError(f"Unknown family: {family}")

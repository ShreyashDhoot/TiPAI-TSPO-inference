"""
policy/tspo_policy.py
─────────────────────
TSPO policy: maps an 8-dim state vector to inpainting knob proposals.

Public API
----------
load_policy(ckpt_path, device)  → TSPOPolicy | None
get_knobs(policy, audit_result, prompt_embed, t_norm, harm_class, n)
    → list[KnobSet]

KnobSet
-------
cfg_scale       float   guidance scale for the inpainter
inversion_depth float   strength parameter (0-1)
seed_offset     int     generator seed
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture constants ────────────────────────────────────────────────────
STATE_DIM        = 8
NUM_CONTINUOUS   = 5
NUM_SEED_BUCKETS = 10

KNOB_BOUNDS = {
    "cfg_scale":       (1.0,  15.0),
    "mask_dilation":   (0.0,   1.0),
    "mask_feather":    (0.0,   1.0),
    "noise_jitter":    (0.0,   0.5),
    "inversion_depth": (0.0,   1.0),
}

HARM_CLASSES = ["safe", "nudity", "violence"]

# vanilla fallback when policy is None
_VANILLA_SWEEPS = [
    (9.0,  0.75,  42),
    (12.0, 0.90, 142),
    (15.0, 1.00, 242),
    (10.0, 0.85, 342),
]


def _denorm(x: float, lo: float, hi: float) -> float:
    return lo + x * (hi - lo)


@dataclass
class KnobSet:
    cfg_scale:       float
    inversion_depth: float
    seed_offset:     int


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

class TSPOPolicy(nn.Module):
    """
    Stochastic policy over continuous inpainting knobs + discrete seed bucket.

    Input : (B, STATE_DIM) state vector
    Output: mean [B, NUM_CONTINUOUS], log_std [B, NUM_CONTINUOUS],
            seed_logits [B, NUM_SEED_BUCKETS]
    """

    def __init__(
        self,
        state_dim:    int   = STATE_DIM,
        hidden_dims:  tuple = (256, 128, 64),
        log_std_min:  float = -4.0,
        log_std_max:  float =  0.5,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers, d = [], state_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
            d = h

        self.trunk        = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(d, NUM_CONTINUOUS)
        self.log_std_head = nn.Linear(d, NUM_CONTINUOUS)
        self.seed_head    = nn.Linear(d, NUM_SEED_BUCKETS)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        return (
            torch.sigmoid(self.mean_head(h)),
            self.log_std_head(h).clamp(self.log_std_min, self.log_std_max),
            self.seed_head(h),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(
    ckpt_path: str | None,
    device: torch.device | str = "cpu",
) -> TSPOPolicy | None:
    """
    Load a TSPOPolicy from a checkpoint file.

    Parameters
    ----------
    ckpt_path : str | None  – path to .pth file; None → returns None (vanilla mode)
    device    : torch.device | str

    Returns
    -------
    TSPOPolicy in eval mode, or None if path is None / missing.
    """
    if ckpt_path is None:
        return None
    if not os.path.exists(ckpt_path):
        print(f"[TSPO] WARNING: checkpoint not found: {ckpt_path}")
        return None

    pol  = TSPOPolicy().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    pol.load_state_dict(state)
    pol.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"[TSPO] Loaded policy from {ckpt_path} (epoch={epoch})")
    return pol


def _build_state(
    audit_result: dict,
    t_norm: float,
    harm_class: str,
    device: torch.device,
) -> torch.Tensor:
    """Build the 8-dim state vector from auditor output."""
    onehot = [0.0, 0.0, 0.0]
    if harm_class in HARM_CLASSES:
        onehot[HARM_CLASSES.index(harm_class)] = 1.0
    else:
        onehot[0] = 1.0   # fallback to 'safe'

    vec = [
        t_norm,
        *onehot,
        audit_result["faithfulness"],
        audit_result["adv_prob"],
        audit_result["seam_quality"],
        0.0,   # reserved slot
    ]
    return torch.tensor(vec, dtype=torch.float32, device=device)


def get_knobs(
    policy:       TSPOPolicy | None,
    audit_result: dict,
    t_norm:       float,
    harm_class:   str,
    n:            int,
    device:       torch.device | str = "cpu",
) -> list[KnobSet]:
    """
    Propose `n` KnobSet candidates.

    When `policy` is None, returns the hardcoded vanilla sweep.

    Parameters
    ----------
    policy       : loaded TSPOPolicy or None
    audit_result : dict returned by AdversarialAuditor.audit_pil()
    t_norm       : timestep normalised to [0,1]
    harm_class   : 'nudity' | 'violence' | 'safe'
    n            : number of candidates to produce
    device       : torch device

    Returns
    -------
    list[KnobSet]  length == n
    """
    if policy is None:
        base = _VANILLA_SWEEPS[:n]
        extra = [(10.0, 0.8, 42 + i * 100) for i in range(max(0, n - len(_VANILLA_SWEEPS)))]
        raw = (base + extra)[:n]
        return [KnobSet(cfg_scale=c, inversion_depth=inv, seed_offset=s) for c, inv, s in raw]

    device = torch.device(device) if isinstance(device, str) else device
    state = _build_state(audit_result, t_norm, harm_class, device)
    state = state.unsqueeze(0).expand(n, -1)   # (n, STATE_DIM)

    with torch.no_grad():
        mean, log_std, seed_logits = policy(state)
        std     = log_std.exp()
        raw_cont = (mean + std * torch.randn_like(mean)).clamp(0.0, 1.0)
        seeds    = torch.distributions.Categorical(
            logits=seed_logits
        ).sample()   # (n,)

    knobs = []
    for i in range(n):
        c = raw_cont[i].tolist()
        knobs.append(KnobSet(
            cfg_scale       = _denorm(c[0], *KNOB_BOUNDS["cfg_scale"]),
            inversion_depth = max(0.05, _denorm(c[4], *KNOB_BOUNDS["inversion_depth"])),
            seed_offset     = 42 + int(seeds[i].item()) * 100,
        ))
    return knobs

"""
policy/tspo_policy.py
─────────────────────
TSPO policy: maps a 257-dim state vector to inpainting knob proposals.

Architecture exactly matches the training checkpoint (TSPO/src/config.py):
  TEXT_DIM  = 512   (auditor LSTM text encoder output)
  PROJ_DIM  = 64
  LATENT_C  = 4
  STATE_DIM = 4 * 64 + 1 = 257

  StateEncoder inputs:
    text_embed  (B, 512)     → text_proj  Linear(512→64)
    latent      (B,4,H,W)    → latent_proj AdaptiveAvgPool2d(4)+Linear(64→64)
    image_embed (B, 256)     → image_proj  Linear(256→64)
    mask_mean   (B, 1)       → mask_proj   Linear(1→64)
    t_norm      (B, 1)       → cat as-is
    ─────────────────────────────────────────────────
    output      (B, 257)

  All embeddings come directly from AdversarialAuditor.audit_pil()
  (img_embed, text_embed fields in AuditResult) — no CLIP needed.

  The StateEncoder weights are saved separately by the training loop:
    state_enc_step{N:05d}.pth  (bare state_dict, not wrapped in a dict)
  Point encoder_checkpoint in config.yaml at the matching step file,
  e.g. weights/state_enc_step00100.pth

Public API
----------
load_policy(ckpt_path, device)         → TSPOPolicy | None
load_state_encoder(ckpt_path, device)  → StateEncoder | None
get_knobs(policy, state, n, device)    → list[KnobSet]
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Architecture constants — must match TSPO/src/config.py ───────────────────
TEXT_DIM         = 512   # auditor SimpleTextEncoder fc output
PROJ_DIM         = 64
LATENT_C         = 4
STATE_DIM        = PROJ_DIM * 4 + 1   # 257
NUM_CONTINUOUS   = 5
NUM_SEED_BUCKETS = 10

KNOB_BOUNDS = {
    "cfg_scale":       (1.0,  15.0),
    "mask_dilation":   (0.0,   1.0),
    "mask_feather":    (0.0,   1.0),
    "noise_jitter":    (0.0,   0.5),
    "inversion_depth": (1,     10),
}

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
    mask_dilation:   float
    mask_feather:    float
    noise_jitter:    float
    inversion_depth: int
    seed_offset:     int


# ─────────────────────────────────────────────────────────────────────────────
# Policy network
# ─────────────────────────────────────────────────────────────────────────────

class TSPOPolicy(nn.Module):
    """Input: (B, 257)  Output: mean[B,5], log_std[B,5], seed_logits[B,10]"""

    def __init__(
        self,
        state_dim:   int   = STATE_DIM,
        hidden_dims: tuple = (256, 128, 64),
        log_std_min: float = -4.0,
        log_std_max: float =  0.5,
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
# StateEncoder — mirrors TSPO/src/models/policy.py exactly
# ─────────────────────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    """
    Encodes auditor outputs into the 257-dim state the policy was trained on.
    No CLIP — all inputs come from AdversarialAuditor.audit_pil().
    """

    def __init__(self):
        super().__init__()
        self.text_proj   = nn.Linear(TEXT_DIM, PROJ_DIM)           # 512 → 64
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(4 * 4 * LATENT_C, PROJ_DIM), nn.ReLU(),     # 64  → 64
        )
        self.image_proj = nn.Linear(256, PROJ_DIM)                  # 256 → 64
        self.mask_proj  = nn.Linear(1,   PROJ_DIM)                  # 1   → 64

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, text_embed, latent, image_embed, mask_mean, t_norm):
        """
        Args:
            text_embed  : (B, 512)     – res_0["text_embed"].unsqueeze(0)
            latent      : (B, 4, H, W) – current VAE latent (float32)
            image_embed : (B, 256)     – res_0["img_embed"].unsqueeze(0)
            mask_mean   : (B, 1)       – mean of binary mask [0,1]
            t_norm      : (B, 1)       – timestep / 1000
        Returns:
            (B, 257)
        """
        p  = F.relu(self.text_proj(text_embed))     # (B, 64)
        z  = F.relu(self.latent_proj(latent))        # (B, 64)
        im = F.relu(self.image_proj(image_embed))    # (B, 64)
        m  = F.relu(self.mask_proj(mask_mean))       # (B, 64)
        return torch.cat([p, z, im, m, t_norm], dim=-1)  # (B, 257)


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(
    ckpt_path: str | None,
    device: "torch.device | str" = "cpu",
) -> "TSPOPolicy | None":
    if ckpt_path is None:
        return None
    if not os.path.exists(ckpt_path):
        print(f"[TSPO] WARNING: policy checkpoint not found: {ckpt_path}")
        return None

    pol  = TSPOPolicy(state_dim=STATE_DIM).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # training saves bare state_dict (policy.state_dict())
    state = ckpt.get("model_state_dict", ckpt)
    pol.load_state_dict(state)
    pol.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"[TSPO] Loaded policy from {ckpt_path} (epoch={epoch})")
    return pol


def load_state_encoder(
    ckpt_path: str | None,
    device: "torch.device | str" = "cpu",
) -> "StateEncoder | None":
    """
    Load StateEncoder from its own checkpoint.

    Training saves it as a bare state_dict:
        torch.save(state_enc.encoder.state_dict(), "state_enc_step00100.pth")

    Set encoder_checkpoint in config.yaml to that file path.
    """
    if ckpt_path is None:
        print("[TSPO] encoder_checkpoint not set in config — running vanilla sweep.")
        return None
    if not os.path.exists(ckpt_path):
        print(f"[TSPO] WARNING: encoder checkpoint not found: {ckpt_path}")
        return None

    enc   = StateEncoder().to(device)
    state = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(state)
    enc.eval()
    print(f"[TSPO] Loaded StateEncoder from {ckpt_path}")
    return enc


# ─────────────────────────────────────────────────────────────────────────────
# Knob generation
# ─────────────────────────────────────────────────────────────────────────────

def get_knobs(
    policy: "TSPOPolicy | None",
    state:  "torch.Tensor | None",
    n:      int,
    device: "torch.device | str" = "cpu",
) -> list[KnobSet]:
    """
    Propose `n` KnobSet candidates.

    Parameters
    ----------
    policy : TSPOPolicy or None → vanilla sweep
    state  : (STATE_DIM,) or (1, STATE_DIM) from StateEncoder. None → vanilla.
    n      : number of candidates
    """
    if policy is None or state is None:
        base  = _VANILLA_SWEEPS[:n]
        extra = [(10.0, 0.8, 42 + i * 100) for i in range(max(0, n - len(_VANILLA_SWEEPS)))]
        return [
            KnobSet(
                cfg_scale=c, mask_dilation=0.5, mask_feather=0.5,
                noise_jitter=0.0,
                inversion_depth=max(1, int(round(inv * 9 + 1))),
                seed_offset=s,
            )
            for c, inv, s in (base + extra)[:n]
        ]

    device = torch.device(device) if isinstance(device, str) else device
    if state.dim() == 1:
        state = state.unsqueeze(0)
    state = state.to(device).expand(n, -1)

    with torch.no_grad():
        mean, log_std, seed_logits = policy(state)
        std      = log_std.exp()
        raw_cont = (mean + std * torch.randn_like(mean)).clamp(0.0, 1.0)
        seeds    = torch.distributions.Categorical(logits=seed_logits).sample()

    knobs = []
    for i in range(n):
        c = raw_cont[i].tolist()
        knobs.append(KnobSet(
            cfg_scale       = _denorm(c[0], *KNOB_BOUNDS["cfg_scale"]),
            mask_dilation   = _denorm(c[1], *KNOB_BOUNDS["mask_dilation"]),
            mask_feather    = _denorm(c[2], *KNOB_BOUNDS["mask_feather"]),
            noise_jitter    = _denorm(c[3], *KNOB_BOUNDS["noise_jitter"]),
            inversion_depth = max(1, int(round(_denorm(c[4], *KNOB_BOUNDS["inversion_depth"])))),
            seed_offset     = int(seeds[i].item()) * 100,
        ))
    return knobs
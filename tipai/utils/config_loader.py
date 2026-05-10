"""
utils/config_loader.py
──────────────────────
Thin wrapper that loads config.yaml and validates required keys.
"""

from __future__ import annotations
import yaml


_REQUIRED_KEYS = [
    "base_sd_model",
    "inpainter_model",
    "auditor_weights",
    "auditor_vocab",
    "total_steps",
    "guidance_scale",
    "audit_steps",
    "n_candidates",
    "delta",
    "tau_P",
    "tau_F",
]


def load_config(path: str = "config.yaml") -> dict:
    """
    Load config.yaml and return as a dict.
    Raises ValueError if required keys are missing.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing required keys: {missing}")

    return cfg

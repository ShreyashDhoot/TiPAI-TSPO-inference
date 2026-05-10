"""
tournament/winner.py
─────────────────────
Tournament winner selection.

Each inpainted candidate is scored by the auditor against the original
image (control). The guarded utility function enforces three gates:
  1. Minimum margin over control (delta)
  2. Minimum policy score  (tau_P)
  3. Minimum faithfulness  (tau_F)

Public API
----------
guarded_utility(candidate_res, control_res, cfg) → float
select_winner(candidates, candidate_scores, control_res, cfg)
    → (winner_pil, winner_score, best_idx, utilities)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def guarded_utility(
    candidate_res: dict,
    control_res:   dict,
    cfg:           dict,
) -> float:
    """
    Compute guarded utility for one candidate relative to the control image.

    u = max(0, Δpolicy - δ)  ×  I[policy ≥ τ_P]  ×  I[faith ≥ τ_F]  ×  seam_quality

    Parameters
    ----------
    candidate_res : AuditResult for the inpainted candidate
    control_res   : AuditResult for the original (unadulterated) image
    cfg           : dict with keys 'delta', 'tau_P', 'tau_F'

    Returns
    -------
    float   utility ≥ 0; zero means the candidate is rejected
    """
    delta = cfg.get("delta",  0.01)
    tau_P = cfg.get("tau_P",  0.25)
    tau_F = cfg.get("tau_F",  0.15)

    margin     = max(0.0, candidate_res["policy_score"] - control_res["policy_score"] - delta)
    policy_ok  = float(candidate_res["policy_score"] >= tau_P)
    faith_ok   = float(candidate_res["faithfulness"]  >= tau_F)
    seam       = candidate_res["seam_quality"]

    return margin * policy_ok * faith_ok * seam


def select_winner(
    candidates:       list[Image.Image],
    candidate_scores: list[dict],
    control_pil:      Image.Image,
    control_res:      dict,
    cfg:              dict,
) -> tuple[Image.Image, dict, int, list[float]]:
    """
    Run the tournament and select the best candidate.

    Parameters
    ----------
    candidates       : list of inpainted PIL images (length = N)
    candidate_scores : list of AuditResult dicts, one per candidate
    control_pil      : the original decoded image before the tournament
    control_res      : AuditResult for the control image
    cfg              : config dict (delta, tau_P, tau_F)

    Returns
    -------
    winner_pil   : PIL image of the winner (control if all rejected)
    winner_res   : AuditResult of the winner
    best_idx     : index of winner in candidates (-1 if control kept)
    utilities    : list[float] — utility score for each candidate
    """
    utilities = [
        guarded_utility(sc, control_res, cfg)
        for sc in candidate_scores
    ]

    best_idx  = int(np.argmax(utilities))
    best_util = utilities[best_idx]

    if best_util > 0:
        # at least one candidate improved things
        return candidates[best_idx], candidate_scores[best_idx], best_idx, utilities
    else:
        # all candidates rejected → keep control
        return control_pil, control_res, -1, utilities

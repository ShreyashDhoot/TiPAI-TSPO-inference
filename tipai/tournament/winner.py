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


# ── ANSI colours for terminal readability ─────────────────────────────────────
_G  = "\033[92m"   # green  — gate passed
_R  = "\033[91m"   # red    — gate failed
_Y  = "\033[93m"   # yellow — header / winner
_C  = "\033[96m"   # cyan   — control row
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset


def _gate(ok: bool) -> str:
    """Return a coloured PASS / FAIL tag."""
    return f"{_G}PASS{_RS}" if ok else f"{_R}FAIL{_RS}"


def guarded_utility(
    candidate_res: dict,
    control_res:   dict,
    cfg:           dict,
) -> tuple[float, dict]:
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
    (utility, gate_info)
        utility   : float ≥ 0; zero means the candidate is rejected
        gate_info : dict with per-gate breakdown for logging / visualisation
    """
    delta = cfg.get("delta",  0.01)
    tau_P = cfg.get("tau_P",  0.25)
    tau_F = cfg.get("tau_F",  0.15)

    raw_margin = candidate_res["policy_score"] - control_res["policy_score"] - delta
    margin     = max(0.0, raw_margin)
    policy_ok  = candidate_res["policy_score"] >= tau_P
    faith_ok   = candidate_res["faithfulness"]  >= tau_F
    adv_ok     = candidate_res["adv_prob"]      <  control_res["adv_prob"]
    seam       = candidate_res["seam_quality"]

    utility = margin * float(policy_ok) * float(faith_ok) * seam

    gate_info = {
        # thresholds used
        "delta":      delta,
        "tau_P":      tau_P,
        "tau_F":      tau_F,
        # raw values
        "raw_margin": raw_margin,
        "margin":     margin,
        "policy":     candidate_res["policy_score"],
        "ctrl_policy":control_res["policy_score"],
        "faith":      candidate_res["faithfulness"],
        "seam":       seam,
        "adv":        candidate_res["adv_prob"],
        "ctrl_adv":   control_res["adv_prob"],
        # gate pass/fail booleans
        "margin_ok":  raw_margin > 0,
        "policy_ok":  policy_ok,
        "faith_ok":   faith_ok,
        "adv_ok":     adv_ok,
    }
    return utility, gate_info


def select_winner(
    candidates:       list[Image.Image],
    candidate_scores: list[dict],
    control_pil:      Image.Image,
    control_res:      dict,
    cfg:              dict,
) -> tuple[Image.Image, dict, int, list[float], list[dict]]:
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
    utilities    : list[float] — guarded utility for each candidate
    gate_infos   : list[dict] — per-gate breakdown for each candidate
    """
    results   = [guarded_utility(sc, control_res, cfg) for sc in candidate_scores]
    utilities = [r[0] for r in results]
    gate_infos = [r[1] for r in results]

    best_idx  = int(np.argmax(utilities))
    best_util = utilities[best_idx]

    # ── debug print ───────────────────────────────────────────────────────────
    gi0 = gate_infos[0]   # all share the same thresholds
    print(
        f"\n  {_B}{_Y}╔══ Tournament Winner Selection "
        f"({len(candidates)} candidates) ══╗{_RS}"
    )
    print(
        f"  {_C}[Control]  "
        f"policy={control_res['policy_score']:.3f}  "
        f"faith={control_res['faithfulness']:.3f}  "
        f"adv={control_res['adv_prob']:.3f}  "
        f"seam={control_res['seam_quality']:.3f}{_RS}"
    )
    print(
        f"  Thresholds → "
        f"δ={gi0['delta']}  τ_P={gi0['tau_P']}  τ_F={gi0['tau_F']}"
    )
    print(f"  {'─'*62}")

    for idx, (sc, u, gi) in enumerate(zip(candidate_scores, utilities, gate_infos)):
        is_best  = (idx == best_idx and best_util > 0)
        tag      = f" {_Y}★ WINNER{_RS}" if is_best else ""
        rejected = "  ← REJECTED" if u == 0 else ""

        print(
            f"  [Cand {idx}]  "
            f"policy={sc['policy_score']:.3f} ({_gate(gi['policy_ok'])} ≥{gi['tau_P']})  "
            f"faith={sc['faithfulness']:.3f} ({_gate(gi['faith_ok'])} ≥{gi['tau_F']})  "
            f"adv={sc['adv_prob']:.3f} ({_gate(gi['adv_ok'])} <ctrl {gi['ctrl_adv']:.3f})  "
            f"seam={sc['seam_quality']:.3f}"
        )
        print(
            f"           "
            f"Δpolicy={gi['raw_margin']:+.4f} ({_gate(gi['margin_ok'])} >δ)  "
            f"margin={gi['margin']:.4f}  "
            f"→ utility={u:.4f}{tag}{rejected}"
        )

    if best_util > 0:
        print(
            f"  {_B}{_Y}╚══ Winner: candidate {best_idx}  "
            f"(utility={best_util:.4f}) ══╝{_RS}\n"
        )
        return candidates[best_idx], candidate_scores[best_idx], best_idx, utilities, gate_infos
    else:
        print(f"  {_B}{_R}╚══ All candidates rejected → keeping control ══╝{_RS}\n")
        return control_pil, control_res, -1, utilities, gate_infos

"""Ensemble adviser: five checkpoints, disagreement-penalized risk ranking.

Preregistration v3 scoring rule (frozen before any ensemble-arm run):

    penalized_risk(c) = risk(c) * (1 + LAMBDA * disagreement(c))

    risk(c)          = trajectory_risk(mean of member predictions)
    disagreement(c)  = mean over the [8, 51] target grid of
                       (std across the 5 members / target_std)
    LAMBDA           = 2.0

Derivation (from the frozen outer holdout, ensemble_eval.json): mean
normalized disagreement is 0.0546 and p95 is 0.0999, so the rule inflates
a candidate's own predicted risk by ~11% typically and ~20% at p95.  The
penalty is proportional to the candidate's own predicted risk: uncertainty
only costs anything when something is at stake, and candidates predicted
harmless (risk ~ 0) are not reordered by disagreement.  Ties resolve to
the earliest candidate in frozen design order, exactly as in v2.

The ensemble mean is used for risk so the scored forecast is the same
quantity that scored 0.1049 NMAE on the outer holdout (better than every
single member).  Member checkpoints: base 873cb77b… plus seeds 20260819
(9bdfd27f…), 20260820 (303c32ed…), 20260821 (618dfdac…), 20260822
(50494734…), all trained on the frozen outer-train split only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aeolus_closed_loop import HistoricalAdviser, trajectory_risk

LAMBDA = 2.0
ENSEMBLE_SEEDS = ("20260819", "20260820", "20260821", "20260822")


class EnsembleAdviser:
    """Ranks catalogue actions by disagreement-penalized ensemble risk."""

    def __init__(self, experiment_dir: Path) -> None:
        members = [experiment_dir / "action-aware-mlp-v1.pt"]
        members += [experiment_dir / "ensemble" / f"seed-{seed}.pt" for seed in ENSEMBLE_SEEDS]
        self.members = [HistoricalAdviser(path) for path in members]
        ref = self.members[0]
        self.target_std = ref.target_std

    def predict_members(
        self, history_f32: np.ndarray, action_f32: np.ndarray | None
    ) -> np.ndarray:
        """Stack member predictions into [K, 8, 51]."""
        return np.stack([m.predict(history_f32, action_f32) for m in self.members])

    def choose(
        self,
        history_f32: np.ndarray,
        candidates: tuple[tuple[str, np.ndarray | None], ...],
    ) -> tuple[str, list[dict[str, Any]]]:
        scored: list[dict[str, Any]] = []
        for candidate_id, action in candidates:
            stack = self.predict_members(history_f32, action)
            risk = trajectory_risk(stack.mean(axis=0))
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = stack.std(axis=0) / np.where(
                    self.target_std == 0, np.nan, self.target_std
                )
            disagreement = float(np.nanmean(normalized))
            penalized = risk * (1.0 + LAMBDA * disagreement)
            scored.append({
                "candidate_id": candidate_id,
                "predicted_risk": risk,
                "disagreement": disagreement,
                "penalized_risk": penalized,
            })
        best = min(range(len(scored)), key=lambda index: scored[index]["penalized_risk"])
        return scored[best]["candidate_id"], scored

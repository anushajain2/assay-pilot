"""Rank variability from bootstrap refits — and what it does not mean.

Refitting a method on bootstrap resamples of the training data and re-ranking
the candidates shows how much the ranking depends on which measurements
happened to be collected. A candidate that is in the top batch under every
refit is a steadier choice than one that drifts in and out.

That is **all** this produces. It is not a calibrated probability of success,
and it is not a prediction interval. Published benchmarking of uncertainty
quantification for protein engineering finds no uncertainty method or
selection policy that wins across datasets and distribution shifts, so
version one of AssayPilot reports rank variability as a diagnostic and does
not select experiments by uncertainty.

Prediction intervals may only be shown as calibrated once held-out coverage
has been measured for the relevant setting; :func:`interval_coverage` is the
check that would have to pass first, and nothing in version one claims it has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..models.base import Method

#: The sentence that must accompany every rank-variability figure.
RANK_VARIABILITY_CAVEAT = (
    "Rank variability across bootstrap refits. This shows how sensitive the ranking is to"
    " the measurements collected so far. It is not a calibrated probability of success and"
    " not a prediction interval."
)


@dataclass
class RankVariability:
    """Per-candidate ranking stability across bootstrap refits."""

    frame: pd.DataFrame
    n_bootstrap: int
    top_k: int
    caveat: str = RANK_VARIABILITY_CAVEAT
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "top_k": self.top_k,
            "caveat": self.caveat,
            "notes": list(self.notes),
        }


def bootstrap_rank_variability(
    method: Method,
    *,
    train_sequences: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    candidate_ids: Sequence[str],
    candidate_sequences: Sequence[str],
    top_k: int,
    n_bootstrap: int = 20,
    seed: int = 0,
) -> RankVariability:
    """Refit ``method`` on bootstrap resamples and summarise candidate ranking stability.

    Resampling is over independent units (distinct measured sequences), so all
    replicates of a sequence are drawn together.

    Returns a frame with, per candidate: mean rank, rank standard deviation,
    and the share of refits in which the candidate landed in the top ``k``.
    """
    train_sequences = list(train_sequences)
    candidate_sequences = list(candidate_sequences)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    ranks = np.full((n_bootstrap, len(candidate_sequences)), np.nan)
    in_top = np.zeros((n_bootstrap, len(candidate_sequences)), dtype=float)
    failures = 0

    for b in range(n_bootstrap):
        drawn = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        rows = np.concatenate([rows_by_group[unique_groups[i]] for i in drawn])
        resampled_sequences = [train_sequences[i] for i in rows]
        resampled_groups = groups[rows]

        try:
            fitted = method.clone()
            fitted.fit(resampled_sequences, y[rows], groups=resampled_groups)
            predictions = np.asarray(fitted.predict(candidate_sequences), dtype=float)
        except Exception:  # pragma: no cover - a degenerate resample, not a user error
            failures += 1
            continue

        order = np.lexsort((np.arange(len(predictions)), -predictions))
        rank_of = np.empty(len(predictions), dtype=float)
        rank_of[order] = np.arange(1, len(predictions) + 1)
        ranks[b] = rank_of
        in_top[b, order[: min(top_k, len(order))]] = 1.0

    with np.errstate(invalid="ignore"):
        mean_rank = np.nanmean(ranks, axis=0)
        std_rank = np.nanstd(ranks, axis=0, ddof=1) if n_bootstrap > 1 else np.zeros(len(mean_rank))
        frequency = np.nanmean(in_top, axis=0)

    frame = pd.DataFrame(
        {
            "candidate_id": list(candidate_ids),
            "mean_rank": mean_rank,
            "rank_std": std_rank,
            "top_k_frequency": frequency,
        }
    )

    notes: list[str] = []
    if failures:
        notes.append(
            f"{failures} of {n_bootstrap} bootstrap refits could not be fitted and were"
            " skipped; the variability figures use the remaining refits."
        )
    return RankVariability(frame=frame, n_bootstrap=n_bootstrap, top_k=int(top_k), notes=notes)


def interval_coverage(
    y_true: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> float:
    """Empirical coverage of prediction intervals on held-out data.

    Provided so that a future release can *measure* calibration before
    claiming it. Nothing in version one displays an interval as calibrated,
    because this check has not been run for the relevant settings.
    """
    true = np.asarray(y_true, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if not (true.shape == low.shape == high.shape):
        raise ValueError("y_true, lower, and upper must have the same shape")
    inside = (true >= low) & (true <= high)
    return float(np.mean(inside)) if len(true) else float("nan")

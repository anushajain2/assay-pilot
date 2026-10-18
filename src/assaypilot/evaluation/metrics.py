"""Metrics chosen to match the decision, not to flatter the model.

The decision is: *given a budget of ``k`` wells, which candidates should be
measured?* Three numbers answer parts of that question, and AssayPilot reports
all three rather than a single headline figure:

``spearman``
    Rank correlation over the whole held-out set — does the method order
    variants correctly at all?
``hit_rate_at_k``
    Of the ``k`` variants the method would have chosen, what share crossed the
    campaign's predefined success threshold? Only available when a threshold
    was defined *before* the evaluation.
``best_at_k``
    The best measured outcome among the ``k`` chosen. This is the number a
    screening campaign actually cares about.

All values follow the convention "higher is better", and all inputs are in
*desirability* space (the pipeline has already flipped the sign for a
minimisation objective).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

#: Every metric here is oriented so that larger values are better.
METRIC_NAMES: tuple[str, ...] = ("spearman", "pearson", "hit_rate_at_k", "best_at_k", "mean_at_k")

HUMAN_NAMES: dict[str, str] = {
    "spearman": "Rank correlation (Spearman)",
    "pearson": "Linear correlation (Pearson)",
    "hit_rate_at_k": "Hit rate in the top batch",
    "best_at_k": "Best measured value in the top batch",
    "mean_at_k": "Mean measured value in the top batch",
}


def _clean_pairs(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} true values vs {pred.shape} predictions")
    finite = np.isfinite(true) & np.isfinite(pred)
    return true[finite], pred[finite]


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Spearman rank correlation, or ``nan`` when it is undefined."""
    true, pred = _clean_pairs(y_true, y_pred)
    if len(true) < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(stats.spearmanr(true, pred).statistic)


def pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Pearson correlation, or ``nan`` when it is undefined."""
    true, pred = _clean_pairs(y_true, y_pred)
    if len(true) < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return float("nan")
    return float(stats.pearsonr(true, pred).statistic)


def top_k_indices(y_pred: Sequence[float], k: int) -> np.ndarray:
    """Indices of the ``k`` highest predictions, ties broken by original order."""
    pred = np.asarray(y_pred, dtype=float)
    k = int(min(max(k, 0), len(pred)))
    if k == 0:
        return np.empty(0, dtype=int)
    # argsort on (-prediction, index) gives a deterministic, stable ranking.
    order = np.lexsort((np.arange(len(pred)), -pred))
    return order[:k]


def hit_rate_at_k(y_pred: Sequence[float], is_hit: Sequence[bool], k: int) -> float:
    """Share of the top-``k`` predicted variants that are hits."""
    hits = np.asarray(is_hit, dtype=bool)
    chosen = top_k_indices(y_pred, k)
    if len(chosen) == 0:
        return float("nan")
    return float(hits[chosen].mean())


def best_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> float:
    """Best measured desirability among the top-``k`` predicted variants."""
    true = np.asarray(y_true, dtype=float)
    chosen = top_k_indices(y_pred, k)
    if len(chosen) == 0:
        return float("nan")
    return float(np.nanmax(true[chosen]))


def mean_at_k(y_true: Sequence[float], y_pred: Sequence[float], k: int) -> float:
    """Mean measured desirability among the top-``k`` predicted variants."""
    true = np.asarray(y_true, dtype=float)
    chosen = top_k_indices(y_pred, k)
    if len(chosen) == 0:
        return float("nan")
    return float(np.nanmean(true[chosen]))


def random_reference(
    y_true: Sequence[float],
    k: int,
    *,
    is_hit: Sequence[bool] | None = None,
    n_draws: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """What random selection of ``k`` variants would have achieved, on average.

    Reported next to every method so that a positive-looking ``best_at_k`` can
    be compared with the number a coin flip would have produced on the same
    held-out set.
    """
    true = np.asarray(y_true, dtype=float)
    n = len(true)
    k = int(min(max(k, 1), n)) if n else 0
    if n == 0:
        return {"best_at_k": float("nan"), "mean_at_k": float("nan"), "hit_rate_at_k": float("nan")}

    rng = np.random.default_rng(seed)
    bests = np.empty(n_draws)
    means = np.empty(n_draws)
    hits = np.empty(n_draws) if is_hit is not None else None
    hit_array = np.asarray(is_hit, dtype=bool) if is_hit is not None else None

    for i in range(n_draws):
        pick = rng.choice(n, size=k, replace=False)
        bests[i] = np.nanmax(true[pick])
        means[i] = np.nanmean(true[pick])
        if hits is not None:
            hits[i] = hit_array[pick].mean()

    return {
        "best_at_k": float(bests.mean()),
        "mean_at_k": float(means.mean()),
        "hit_rate_at_k": float(hits.mean()) if hits is not None else float("nan"),
    }


@dataclass
class MetricValues:
    """Metric values for one method on one split."""

    values: dict[str, float] = field(default_factory=dict)
    k: int = 0
    n_test: int = 0

    def get(self, name: str) -> float:
        return float(self.values.get(name, float("nan")))

    def to_dict(self) -> dict[str, Any]:
        return {"k": self.k, "n_test": self.n_test, **{k: v for k, v in self.values.items()}}


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    k: int,
    is_hit: Sequence[bool] | None = None,
) -> MetricValues:
    """Compute every applicable metric for one method on one split."""
    true = np.asarray(y_true, dtype=float)
    values = {
        "spearman": spearman(true, y_pred),
        "pearson": pearson(true, y_pred),
        "best_at_k": best_at_k(true, y_pred, k),
        "mean_at_k": mean_at_k(true, y_pred, k),
    }
    values["hit_rate_at_k"] = (
        hit_rate_at_k(y_pred, is_hit, k) if is_hit is not None else float("nan")
    )
    return MetricValues(values=values, k=int(k), n_test=int(len(true)))


def choose_primary_metric(*, has_threshold: bool, n_test: int, k: int) -> str:
    """Pick the metric the recommendation is based on.

    Hit rate is the most decision-shaped metric, but it is unusable when no
    success threshold was defined in advance, and unstable when the top batch
    is a large share of a small held-out set. In those cases rank correlation
    over the full held-out set is the more reliable basis for a recommendation.
    """
    if has_threshold and n_test >= 2 * max(k, 1):
        return "hit_rate_at_k"
    return "spearman"

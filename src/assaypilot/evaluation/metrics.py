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
from typing import Any, NamedTuple

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


#: Above this share of measured variants clearing the success threshold, hit
#: rate stops separating methods: random selection already scores near the base
#: rate and the ceiling is 1, so every method lands in a narrow band.
MAX_HIT_BASE_RATE = 0.25

#: Below this many hits in the smallest held-out partition, hit rate is mostly
#: noise — a single variant moving in or out swings it.
MIN_HITS_IN_TEST = 5


class MetricChoice(NamedTuple):
    """Which metric drives the decision, and why that one."""

    metric: str
    reason: str
    hit_base_rate: float | None = None

    @property
    def fell_back(self) -> bool:
        """True when a success threshold existed but hit rate was not usable."""
        return self.metric != "hit_rate_at_k" and self.hit_base_rate is not None


def choose_primary_metric(
    *,
    has_threshold: bool,
    n_test: int,
    k: int,
    hit_base_rate: float | None = None,
    n_hits_in_test: int | None = None,
) -> MetricChoice:
    """Pick the metric the recommendation is based on, and say why.

    Hit rate is the most decision-shaped metric, but it only separates methods
    under conditions worth stating:

    * a success threshold has to exist, and to have been declared in advance;
    * the held-out set has to be large enough relative to the batch, or the
      top batch is most of it;
    * the threshold has to isolate a *top slice*. If most measured variants
      already clear it — which happens when values pile up at a ceiling, or
      when the threshold was set optimistically — then random selection scores
      near the base rate too, and the number cannot distinguish anything;
    * there have to be enough hits held out to count.

    When any of those fail, rank correlation over the whole held-out set is
    the more reliable basis, and the returned reason says which one failed so
    the report can repeat it.
    """
    if not has_threshold:
        return MetricChoice(
            "spearman",
            "no success threshold is defined for this campaign, so no hit rate can be"
            " computed; the ranking of all held-out variants is used instead",
        )
    if n_test < 2 * max(k, 1):
        return MetricChoice(
            "spearman",
            f"the batch of {k} is more than half the smallest held-out partition"
            f" ({n_test} measurements), so a hit rate over the top batch would be"
            " near-constant; the ranking of all held-out variants is used instead",
            hit_base_rate,
        )
    if hit_base_rate is not None and hit_base_rate > MAX_HIT_BASE_RATE:
        return MetricChoice(
            "spearman",
            f"{hit_base_rate:.0%} of the measured variants already clear the success"
            " threshold, so a hit rate cannot separate methods — selecting at random"
            f" would score about {hit_base_rate:.0%} too. The ranking of all held-out"
            " variants is used instead. Consider a stricter threshold if hit rate is the"
            " number you care about.",
            hit_base_rate,
        )
    if n_hits_in_test is not None and n_hits_in_test < MIN_HITS_IN_TEST:
        return MetricChoice(
            "spearman",
            f"only {n_hits_in_test} held-out measurement(s) clear the success threshold,"
            f" fewer than the {MIN_HITS_IN_TEST} needed for a hit rate to be more than"
            " noise; the ranking of all held-out variants is used instead",
            hit_base_rate,
        )
    return MetricChoice(
        "hit_rate_at_k",
        f"a success threshold is defined and {hit_base_rate:.0%} of measured variants clear"
        " it, so the share of the top batch that are hits is both computable and"
        " discriminating"
        if hit_base_rate is not None
        else "a success threshold is defined and the held-out set is large enough for a hit"
        " rate over the top batch",
        hit_base_rate,
    )

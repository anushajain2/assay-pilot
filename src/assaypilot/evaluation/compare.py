"""Running the comparison suite and deciding what it supports.

This module answers two questions and keeps them separate:

1. *How did each method score on the held-out examples?* — per-split metrics,
   their spread across splits, and a paired bootstrap confidence interval for
   the difference between methods.
2. *What does that justify doing?* — a recommendation produced by one fixed,
   stated rule.

The rule, in full:

* Start from the **simplest available method** as the incumbent.
* Walk the remaining methods in increasing complexity. Replace the incumbent
  only when the paired bootstrap interval for ``candidate - incumbent`` on the
  primary metric lies entirely above zero.
* If nothing clears that bar, keep the simplest method and say the methods
  were **not distinguishable**.

Two consequences are deliberate. A small numerical lead never becomes a
"winner". And when the evidence is thin — only a random split, few held-out
sequences, or an interval that straddles zero — the result is labelled
``exploratory`` rather than being reported as a validated recommendation.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..campaign import Campaign
from ..data.loaders import MeasurementSet
from ..models.base import Method, MethodInfo
from .metrics import (
    HUMAN_NAMES,
    MetricValues,
    choose_primary_metric,
    compute_metrics,
    random_reference,
)
from .splits import Split, SplitPlan

logger = logging.getLogger(__name__)

EvidenceLevel = str  # "supported" | "exploratory" | "none"

#: Below this many distinct held-out sequences, a comparison is exploratory
#: whatever the numbers say.
MIN_TEST_SEQUENCES_FOR_SUPPORT = 20


@dataclass
class SplitPrediction:
    """One method's predictions on one split's held-out rows."""

    split_name: str
    method_key: str
    test_index: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray
    is_hit: np.ndarray | None
    metrics: MetricValues
    fit_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodSummary:
    """Everything the report shows about one method."""

    info: MethodInfo
    per_split: dict[str, MetricValues]
    mean_metrics: dict[str, float]
    spread_metrics: dict[str, float]
    fit_details: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.info.key

    def mean(self, metric: str) -> float:
        return float(self.mean_metrics.get(metric, float("nan")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.info.to_dict(),
            "mean_metrics": self.mean_metrics,
            "spread_across_splits": self.spread_metrics,
            "per_split": {name: m.to_dict() for name, m in self.per_split.items()},
            "fit_details": self.fit_details,
            "notes": list(self.notes),
        }


@dataclass
class PairedComparison:
    """Paired bootstrap comparison of two methods on the primary metric."""

    method_key: str
    reference_key: str
    metric: str
    difference: float
    ci_low: float
    ci_high: float
    fraction_better: float
    n_bootstrap: int

    @property
    def distinguishable(self) -> bool:
        """True when the whole interval lies above zero."""
        return bool(np.isfinite(self.ci_low) and self.ci_low > 0)

    @property
    def sentence(self) -> str:
        if not np.isfinite(self.difference):
            return (
                f"{self.method_key} could not be compared with {self.reference_key} on"
                f" {HUMAN_NAMES.get(self.metric, self.metric)}."
            )
        verdict = (
            "a difference supported by the resampling"
            if self.distinguishable
            else ("not distinguishable from the reference within the resampling interval")
        )
        return (
            f"{HUMAN_NAMES.get(self.metric, self.metric)}: {self.difference:+.3f} versus"
            f" {self.reference_key} (95% interval {self.ci_low:+.3f} to {self.ci_high:+.3f}),"
            f" {verdict}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method_key,
            "reference": self.reference_key,
            "metric": self.metric,
            "difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "fraction_better": self.fraction_better,
            "n_bootstrap": self.n_bootstrap,
            "distinguishable": self.distinguishable,
        }


@dataclass
class Recommendation:
    """Which method the evidence supports using, and how strongly."""

    method_key: str
    evidence_level: EvidenceLevel
    rule: str
    reasons: list[str]
    distinguishable: bool
    primary_metric: str

    @property
    def is_exploratory(self) -> bool:
        return self.evidence_level != "supported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method_key,
            "evidence_level": self.evidence_level,
            "rule": self.rule,
            "reasons": list(self.reasons),
            "distinguishable": self.distinguishable,
            "primary_metric": self.primary_metric,
        }


@dataclass
class EvaluationResult:
    """The complete model-evidence section of a run."""

    campaign: Campaign
    split_plan: SplitPlan
    primary_metric: str
    k: int
    summaries: dict[str, MethodSummary]
    comparisons: list[PairedComparison]
    recommendation: Recommendation
    random_reference: dict[str, float]
    unavailable: dict[str, str] = field(default_factory=dict)
    predictions: list[SplitPrediction] = field(default_factory=list)
    n_bootstrap: int = 1000
    seed: int = 0

    @property
    def recommended_summary(self) -> MethodSummary:
        return self.summaries[self.recommendation.method_key]

    def comparison_for(self, method_key: str) -> PairedComparison | None:
        for comparison in self.comparisons:
            if comparison.method_key == method_key:
                return comparison
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_plan": self.split_plan.to_dict(),
            "primary_metric": self.primary_metric,
            "k": self.k,
            "methods": {k: s.to_dict() for k, s in self.summaries.items()},
            "comparisons": [c.to_dict() for c in self.comparisons],
            "recommendation": self.recommendation.to_dict(),
            "random_reference": self.random_reference,
            "unavailable_methods": dict(self.unavailable),
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }


def _hits_for(
    campaign: Campaign, values: np.ndarray, *, threshold: float | None
) -> np.ndarray | None:
    """Hit flags in desirability space, or ``None`` when no threshold applies.

    ``threshold`` is the campaign threshold already mapped into the value space
    the measurements are in (see
    :meth:`~assaypilot.data.loaders.MeasurementSet.threshold_in_value_space`),
    because a log-transformed measurement must not be compared against a raw
    threshold.
    """
    if threshold is None:
        return None
    return np.asarray(values, dtype=float) >= campaign.sign * threshold


def _fit_and_predict(
    method: Method,
    measurements: MeasurementSet,
    split: Split,
    *,
    sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a fresh clone on the split's training rows and predict its test rows."""
    frame = measurements.frame
    train = frame.iloc[split.train_index]
    test = frame.iloc[split.test_index]

    fitted = method.clone()
    fitted.fit(
        train["sequence"].tolist(),
        sign * train["value"].to_numpy(dtype=float),
        groups=train["seq_hash"].to_numpy(),
    )
    y_pred = np.asarray(fitted.predict(test["sequence"].tolist()), dtype=float)
    y_true = sign * test["value"].to_numpy(dtype=float)
    groups = test["seq_hash"].to_numpy()
    return y_true, y_pred, groups, fitted.fit_details()


def _bootstrap_metric_samples(
    predictions: dict[str, SplitPrediction],
    *,
    metric: str,
    k: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Paired bootstrap over independent units (distinct held-out sequences).

    The same resampled unit indices are used for every method, which is what
    makes the comparison paired: differences are not inflated by resampling
    noise that both methods share.
    """
    any_prediction = next(iter(predictions.values()))
    groups = any_prediction.groups
    unique_groups = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}

    rng = np.random.default_rng(seed)
    samples = {key: np.full(n_bootstrap, np.nan) for key in predictions}

    for b in range(n_bootstrap):
        drawn = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        rows = np.concatenate([rows_by_group[unique_groups[i]] for i in drawn])
        for key, prediction in predictions.items():
            is_hit = prediction.is_hit[rows] if prediction.is_hit is not None else None
            values = compute_metrics(
                prediction.y_true[rows], prediction.y_pred[rows], k=k, is_hit=is_hit
            )
            samples[key][b] = values.get(metric)
    return samples


def _paired_comparisons(
    per_split_samples: list[dict[str, np.ndarray]],
    *,
    reference_key: str,
    metric: str,
    n_bootstrap: int,
) -> list[PairedComparison]:
    """Average paired bootstrap differences across splits and form intervals."""
    if not per_split_samples:
        return []

    keys = [k for k in per_split_samples[0] if k != reference_key]
    comparisons: list[PairedComparison] = []

    for key in keys:
        diffs = []
        for samples in per_split_samples:
            if key not in samples or reference_key not in samples:
                continue
            diffs.append(samples[key] - samples[reference_key])
        if not diffs:
            continue
        stacked = np.nanmean(np.vstack(diffs), axis=0)
        finite = stacked[np.isfinite(stacked)]
        if finite.size == 0:
            comparisons.append(
                PairedComparison(
                    method_key=key,
                    reference_key=reference_key,
                    metric=metric,
                    difference=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    fraction_better=float("nan"),
                    n_bootstrap=n_bootstrap,
                )
            )
            continue
        comparisons.append(
            PairedComparison(
                method_key=key,
                reference_key=reference_key,
                metric=metric,
                difference=float(np.mean(finite)),
                ci_low=float(np.percentile(finite, 2.5)),
                ci_high=float(np.percentile(finite, 97.5)),
                fraction_better=float(np.mean(finite > 0)),
                n_bootstrap=n_bootstrap,
            )
        )
    return comparisons


def _chance_level(primary_metric: str, random_reference: dict[str, float]) -> float:
    """The value the primary metric would take without a useful model.

    Zero for a rank correlation; for the decision-shaped metrics it is what
    random selection of the same number of variants achieved on the same
    held-out set. Returns ``nan`` when no reference is available, in which case
    the caller makes no claim either way.
    """
    if primary_metric == "spearman":
        return 0.0
    reference = random_reference.get(primary_metric, float("nan"))
    return float(reference)


def _decide(
    *,
    summaries: dict[str, MethodSummary],
    ordered_keys: list[str],
    comparisons_by_pair: dict[tuple[str, str], PairedComparison],
    primary_metric: str,
    split_plan: SplitPlan,
    n_test_groups: int,
    random_reference: dict[str, float],
) -> Recommendation:
    """Apply the stated selection rule and set the evidence level."""
    rule = (
        "Start from the simplest available method. Adopt a more complex method only when the"
        " paired bootstrap 95% interval for its improvement on the primary metric lies"
        " entirely above zero. Otherwise keep the simpler method."
    )
    reasons: list[str] = []
    incumbent = ordered_keys[0]
    distinguishable = False

    for key in ordered_keys[1:]:
        comparison = comparisons_by_pair.get((key, incumbent))
        if comparison is None:
            reasons.append(f"{key} was not compared against {incumbent}; incumbent kept.")
            continue
        if comparison.distinguishable:
            reasons.append(f"{key} improved on {incumbent}: {comparison.sentence}")
            incumbent = key
            distinguishable = True
        else:
            reasons.append(f"{key} did not improve on {incumbent}: {comparison.sentence}")

    if not distinguishable:
        reasons.append(
            "No method was distinguishable from the simplest supported method within the"
            " resampling interval, so the simpler method is used."
        )

    evidence: EvidenceLevel = "supported"
    if split_plan.is_interpolation_only:
        evidence = "exploratory"
        reasons.append(
            "Only a random (interpolation) split was available, so these results are not"
            " evidence about candidates further from the measured variants."
        )
    if n_test_groups < MIN_TEST_SEQUENCES_FOR_SUPPORT:
        evidence = "exploratory"
        reasons.append(
            f"Only {n_test_groups} distinct held-out sequence(s) were available"
            f" (fewer than {MIN_TEST_SEQUENCES_FOR_SUPPORT}), which is too few to separate"
            " methods reliably."
        )

    summary = summaries[incumbent]
    primary_value = summary.mean(primary_metric)
    if not np.isfinite(primary_value):
        evidence = "exploratory"
        reasons.append(
            f"The primary metric ({HUMAN_NAMES.get(primary_metric, primary_metric)}) could not"
            " be computed for the selected method."
        )
    else:
        # "Better than chance" means something different for each metric, and the
        # check has to run for whichever one drove the decision: a campaign with a
        # success threshold uses hit rate, and skipping the check there would let a
        # method that orders variants backwards be reported as supported.
        chance = _chance_level(primary_metric, random_reference)
        if np.isfinite(chance) and primary_value <= chance:
            evidence = "exploratory"
            reasons.append(
                "The selected method did not beat chance on the primary metric"
                f" ({HUMAN_NAMES.get(primary_metric, primary_metric)}"
                f" {primary_value:.3f} against a chance level of {chance:.3f}),"
                " so the resulting batch is exploratory."
            )

    return Recommendation(
        method_key=incumbent,
        evidence_level=evidence,
        rule=rule,
        reasons=reasons,
        distinguishable=distinguishable,
        primary_metric=primary_metric,
    )


def evaluate_methods(
    methods: Sequence[Method],
    measurements: MeasurementSet,
    split_plan: SplitPlan,
    *,
    campaign: Campaign | None = None,
    k: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    unavailable: dict[str, str] | None = None,
) -> EvaluationResult:
    """Evaluate every method on every split and apply the selection rule.

    All methods see exactly the same training rows and are scored on exactly
    the same held-out rows, split by split.
    """
    campaign = campaign or measurements.campaign
    if not methods:
        raise ValueError("no methods to evaluate")
    if not split_plan.splits:
        raise ValueError("the split plan contains no splits")

    sign = campaign.sign
    batch_size = k if k is not None else campaign.selection.batch_size
    # The threshold is mapped into the measurements' value space once, so a
    # transformed assay cannot silently redefine what counts as a hit.
    threshold = measurements.threshold_in_value_space()

    ordered = sorted(methods, key=lambda m: (m.info.complexity, m.info.key))
    ordered_keys = [m.info.key for m in ordered]

    # The primary metric and the evaluation budget are fixed before any model is
    # fitted, so that no choice here can depend on the results it will judge.
    #
    # One budget is used for every split. Scoring a 10-row fold at k=10 and a
    # 40-row fold at k=24 and then averaging would produce a mean of two
    # different quantities, labelled with whichever k happened to be smaller.
    k_used = int(min([batch_size] + [max(1, s.n_test) for s in split_plan.splits]))
    # The "is the top batch a large share of the held-out set?" guard has to be
    # checked per split: a pooled row count across splits can clear the bar
    # while every individual fold is almost entirely top-batch.
    smallest_test = min(s.n_test for s in split_plan.splits)
    primary_metric = choose_primary_metric(
        has_threshold=threshold is not None,
        n_test=smallest_test,
        k=k_used,
    )

    per_split_metrics: dict[str, dict[str, MetricValues]] = {key: {} for key in ordered_keys}
    fit_details: dict[str, dict[str, Any]] = {key: {} for key in ordered_keys}
    all_predictions: list[SplitPrediction] = []
    per_split_samples: list[dict[str, np.ndarray]] = []
    random_refs: list[dict[str, float]] = []
    test_groups: set[str] = set()

    for split in split_plan.splits:
        split_predictions: dict[str, SplitPrediction] = {}

        for method in ordered:
            y_true, y_pred, groups, details = _fit_and_predict(
                method, measurements, split, sign=sign
            )
            is_hit = _hits_for(campaign, y_true, threshold=threshold)
            metrics = compute_metrics(y_true, y_pred, k=k_used, is_hit=is_hit)
            prediction = SplitPrediction(
                split_name=split.name,
                method_key=method.info.key,
                test_index=np.asarray(split.test_index),
                y_true=y_true,
                y_pred=y_pred,
                groups=groups,
                is_hit=is_hit,
                metrics=metrics,
                fit_details=details,
            )
            split_predictions[method.info.key] = prediction
            all_predictions.append(prediction)
            per_split_metrics[method.info.key][split.name] = metrics
            fit_details[method.info.key] = details

        reference = next(iter(split_predictions.values()))
        test_groups.update(reference.groups.tolist())
        random_refs.append(
            random_reference(
                reference.y_true,
                k_used,
                is_hit=reference.is_hit,
                # crc32, not hash(): str hashing is salted per process, which
                # would make the random reference irreproducible.
                seed=seed + zlib.crc32(split.name.encode("utf-8")) % 10_000,
            )
        )

        per_split_samples.append(
            _bootstrap_metric_samples(
                split_predictions,
                metric=primary_metric,
                k=k_used,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )

    summaries: dict[str, MethodSummary] = {}
    for method in ordered:
        key = method.info.key
        by_split = per_split_metrics[key]
        mean_metrics: dict[str, float] = {}
        spread_metrics: dict[str, float] = {}
        for metric_name in next(iter(by_split.values())).values:
            series = np.array([m.get(metric_name) for m in by_split.values()], dtype=float)
            finite = series[np.isfinite(series)]
            mean_metrics[metric_name] = float(finite.mean()) if finite.size else float("nan")
            spread_metrics[metric_name] = (
                float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            )
        summaries[key] = MethodSummary(
            info=method.info,
            per_split=by_split,
            mean_metrics=mean_metrics,
            spread_metrics=spread_metrics,
            fit_details=fit_details[key],
        )

    reference_key = ordered_keys[0]
    comparisons = _paired_comparisons(
        per_split_samples,
        reference_key=reference_key,
        metric=primary_metric,
        n_bootstrap=n_bootstrap,
    )
    comparisons_by_pair = {(c.method_key, c.reference_key): c for c in comparisons}

    # Comparisons against an adopted incumbent other than the simplest method
    # need their own paired intervals.
    for incumbent in ordered_keys[1:-1]:
        extra = _paired_comparisons(
            per_split_samples,
            reference_key=incumbent,
            metric=primary_metric,
            n_bootstrap=n_bootstrap,
        )
        for comparison in extra:
            comparisons_by_pair.setdefault(
                (comparison.method_key, comparison.reference_key), comparison
            )

    pooled_random: dict[str, float] = {}
    for metric in ("best_at_k", "mean_at_k", "hit_rate_at_k"):
        values = np.array([ref[metric] for ref in random_refs], dtype=float)
        finite = values[np.isfinite(values)]
        # hit_rate is nan throughout when no success threshold was declared.
        pooled_random[metric] = float(finite.mean()) if finite.size else float("nan")

    recommendation = _decide(
        summaries=summaries,
        ordered_keys=ordered_keys,
        comparisons_by_pair=comparisons_by_pair,
        primary_metric=primary_metric,
        split_plan=split_plan,
        n_test_groups=len(test_groups),
        random_reference=pooled_random,
    )

    return EvaluationResult(
        campaign=campaign,
        split_plan=split_plan,
        primary_metric=primary_metric,
        k=k_used,
        summaries=summaries,
        comparisons=list(comparisons_by_pair.values()),
        recommendation=recommendation,
        random_reference=pooled_random,
        unavailable=dict(unavailable or {}),
        predictions=all_predictions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

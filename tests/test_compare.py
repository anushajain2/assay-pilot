"""Metrics, paired comparison, and the rule that picks a method.

The behaviour that matters most: a small numerical lead must not become a
recommendation, and weak evidence must be labelled as weak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assaypilot import load_measurements
from assaypilot.evaluation.compare import evaluate_methods
from assaypilot.evaluation.metrics import (
    best_at_k,
    choose_primary_metric,
    compute_metrics,
    hit_rate_at_k,
    random_reference,
    spearman,
    top_k_indices,
)
from assaypilot.evaluation.splits import choose_split_plan
from assaypilot.models.base import Method, MethodInfo


class ConstantMethod(Method):
    """Predicts a fixed vector, so a comparison's outcome can be controlled exactly."""

    def __init__(self, key: str, complexity: int, scorer, *, uses_labels: bool = True) -> None:
        self._key = key
        self._complexity = complexity
        self._scorer = scorer
        self._uses_labels = uses_labels

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key=self._key,
            name=self._key,
            kind="baseline" if self._complexity == 1 else "assay_trained",
            description="test double",
            uses_labels=self._uses_labels,
            complexity=self._complexity,
        )

    def clone(self) -> ConstantMethod:
        return ConstantMethod(
            self._key, self._complexity, self._scorer, uses_labels=self._uses_labels
        )

    def fit(self, sequences, y, *, groups=None):  # noqa: ANN001
        self._truth = {s: v for s, v in zip(sequences, y, strict=True)}
        return self

    def predict(self, sequences):  # noqa: ANN001
        return np.asarray([self._scorer(s) for s in sequences], dtype=float)


# ------------------------------------------------------------------- metrics


def test_spearman_handles_degenerate_input():
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))  # fewer than three points
    assert np.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))  # no variation in truth
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_top_k_indices_are_deterministic_under_ties():
    predictions = [1.0, 1.0, 1.0, 0.0]
    assert list(top_k_indices(predictions, 2)) == [0, 1]
    assert list(top_k_indices(predictions, 10)) == [0, 1, 2, 3]
    assert list(top_k_indices(predictions, 0)) == []


def test_hit_rate_and_best_at_k():
    y_true = [0.0, 5.0, 1.0, 4.0]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    is_hit = [False, True, False, True]
    assert hit_rate_at_k(y_pred, is_hit, 2) == pytest.approx(1.0)
    assert best_at_k(y_true, y_pred, 2) == pytest.approx(5.0)
    assert best_at_k(y_true, y_pred, 1) == pytest.approx(5.0)


def test_random_reference_is_between_min_and_max():
    y_true = list(np.linspace(0, 10, 50))
    reference = random_reference(y_true, 5, n_draws=200, seed=0)
    assert 0 < reference["best_at_k"] < 10
    assert 0 < reference["mean_at_k"] < 10


def test_primary_metric_requires_a_predeclared_threshold():
    assert choose_primary_metric(has_threshold=False, n_test=100, k=10) == "spearman"
    assert choose_primary_metric(has_threshold=True, n_test=100, k=10) == "hit_rate_at_k"
    # Too few held-out examples relative to the batch: hit rate would be unstable.
    assert choose_primary_metric(has_threshold=True, n_test=12, k=10) == "spearman"


def test_compute_metrics_reports_nan_without_hit_flags():
    values = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], k=2)
    assert np.isnan(values.get("hit_rate_at_k"))
    assert values.get("spearman") == pytest.approx(1.0)


# --------------------------------------------------------------- the decision


def _measurements_with_rounds(campaign, n=90, seed=0):
    rng = np.random.default_rng(seed)
    reference = campaign.reference_sequence
    positions = [3, 5, 7, 9, 11, 13, 15, 17]
    rows = []
    seen = set()
    while len(rows) < n:
        n_subs = int(rng.integers(1, 4))
        chosen = rng.choice(positions, size=n_subs, replace=False)
        residues = list(reference)
        for position in chosen:
            residues[position - 1] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
        sequence = "".join(residues)
        if sequence in seen:
            continue
        seen.add(sequence)
        rows.append(
            {
                "variant_id": f"v{len(rows):03d}",
                "sequence": sequence,
                "value": float(rng.normal()),
                "condition_id": campaign.condition_id,
                "round_id": f"R{(len(rows) % 3) + 1}",
            }
        )
    return load_measurements(pd.DataFrame(rows), campaign, strict=False)


def test_indistinguishable_methods_keep_the_simpler_one(campaign):
    measurements = _measurements_with_rounds(campaign)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    rng = np.random.default_rng(0)

    simple = ConstantMethod("simple", 1, lambda s: truth[s] + rng.normal(0, 1.0))
    complex_ = ConstantMethod("complex", 3, lambda s: truth[s] + rng.normal(0, 1.0))

    result = evaluate_methods(
        [simple, complex_], measurements, plan, campaign=campaign, k=5, n_bootstrap=100, seed=0
    )
    assert result.recommendation.method_key == "simple"
    assert result.recommendation.distinguishable is False
    assert any(
        "not distinguishable" in r or "did not improve" in r for r in result.recommendation.reasons
    )


def test_a_clearly_better_method_is_adopted(campaign):
    measurements = _measurements_with_rounds(campaign)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    rng = np.random.default_rng(1)

    noise = ConstantMethod("simple", 1, lambda s: rng.normal())
    oracle = ConstantMethod("complex", 3, lambda s: truth[s])

    result = evaluate_methods(
        [noise, oracle], measurements, plan, campaign=campaign, k=5, n_bootstrap=200, seed=0
    )
    assert result.recommendation.method_key == "complex"
    assert result.recommendation.distinguishable is True
    comparison = result.comparison_for("complex")
    assert comparison is not None and comparison.distinguishable
    assert comparison.ci_low > 0


def test_random_split_only_is_labelled_exploratory(campaign):
    measurements = _measurements_with_rounds(campaign)
    measurements.frame["round_id"] = None
    plan = choose_split_plan(measurements, policy="random")
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    oracle = ConstantMethod("oracle", 1, lambda s: truth[s])

    result = evaluate_methods(
        [oracle], measurements, plan, campaign=campaign, k=5, n_bootstrap=50, seed=0
    )
    assert result.recommendation.evidence_level == "exploratory"
    assert any("interpolation" in reason for reason in result.recommendation.reasons)


def test_unavailable_methods_are_reported_not_hidden(campaign):
    measurements = _measurements_with_rounds(campaign)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    method = ConstantMethod("simple", 1, lambda s: truth[s])

    result = evaluate_methods(
        [method],
        measurements,
        plan,
        campaign=campaign,
        k=5,
        n_bootstrap=20,
        seed=0,
        unavailable={"pretrained": "torch is not installed"},
    )
    assert result.unavailable == {"pretrained": "torch is not installed"}
    assert "pretrained" in result.to_dict()["unavailable_methods"]


def test_evaluation_is_reproducible(campaign):
    measurements = _measurements_with_rounds(campaign)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))

    def run():
        methods = [
            ConstantMethod("simple", 1, lambda s: truth[s] * 0.5),
            ConstantMethod("complex", 3, lambda s: truth[s]),
        ]
        return evaluate_methods(
            methods, measurements, plan, campaign=campaign, k=5, n_bootstrap=100, seed=42
        )

    first, second = run(), run()
    assert first.recommendation.method_key == second.recommendation.method_key
    for key in first.summaries:
        np.testing.assert_allclose(
            list(first.summaries[key].mean_metrics.values()),
            list(second.summaries[key].mean_metrics.values()),
            equal_nan=True,
        )
    assert [c.ci_low for c in first.comparisons] == [c.ci_low for c in second.comparisons]


def test_evaluate_requires_methods_and_splits(campaign):
    measurements = _measurements_with_rounds(campaign)
    plan = choose_split_plan(measurements)
    with pytest.raises(ValueError, match="no methods"):
        evaluate_methods([], measurements, plan, campaign=campaign)
    plan.splits = []
    with pytest.raises(ValueError, match="no splits"):
        evaluate_methods(
            [ConstantMethod("simple", 1, lambda s: 0.0)], measurements, plan, campaign=campaign
        )

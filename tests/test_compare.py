"""Metrics, paired comparison, and the rule that picks a method.

The behaviour that matters most: a small numerical lead must not become a
recommendation, and weak evidence must be labelled as weak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assaypilot import load_measurements
from assaypilot.evaluation.compare import PairedComparison, evaluate_methods
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
    no_threshold = choose_primary_metric(has_threshold=False, n_test=100, k=10)
    assert no_threshold.metric == "spearman"
    assert "no success threshold" in no_threshold.reason

    usable = choose_primary_metric(
        has_threshold=True, n_test=100, k=10, hit_base_rate=0.1, n_hits_in_test=10
    )
    assert usable.metric == "hit_rate_at_k"

    # Too few held-out examples relative to the batch: hit rate would be unstable.
    crowded = choose_primary_metric(has_threshold=True, n_test=12, k=10)
    assert crowded.metric == "spearman"
    assert "more than half" in crowded.reason


def test_a_threshold_most_variants_clear_cannot_drive_the_decision():
    """The CcdB case: if 79% of measured variants are hits, so is a random batch."""
    choice = choose_primary_metric(
        has_threshold=True, n_test=200, k=24, hit_base_rate=0.79, n_hits_in_test=150
    )
    assert choice.metric == "spearman"
    assert choice.fell_back
    assert "79% of the measured variants already clear" in choice.reason
    assert "stricter threshold" in choice.reason


def test_too_few_held_out_hits_cannot_drive_the_decision():
    choice = choose_primary_metric(
        has_threshold=True, n_test=200, k=24, hit_base_rate=0.01, n_hits_in_test=2
    )
    assert choice.metric == "spearman"
    assert choice.fell_back
    assert "only 2 held-out measurement(s) clear" in choice.reason


def test_a_usable_threshold_is_reported_as_such():
    choice = choose_primary_metric(
        has_threshold=True, n_test=200, k=24, hit_base_rate=0.1, n_hits_in_test=20
    )
    assert choice.metric == "hit_rate_at_k"
    assert not choice.fell_back
    assert "10% of measured variants clear it" in choice.reason


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


def _threshold_campaign(campaign, threshold: float):
    return campaign.__class__.from_dict({**campaign.to_dict(), "success_threshold": threshold})


def test_a_backwards_method_is_never_supported_under_a_hit_rate_metric(campaign):
    """The better-than-chance gate must apply to whichever metric drove the decision.

    With a usable success threshold the primary metric is hit rate, and a
    method that orders variants perfectly backwards must not be reported as
    supported.
    """
    measurements = _measurements_with_rounds(campaign, n=180, seed=3)
    # The 80th percentile: high enough to be a real bar, low enough that each
    # held-out round contains enough hits for the metric to be usable.
    threshold = float(measurements.frame["value"].quantile(0.8))
    with_threshold = _threshold_campaign(campaign, threshold)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))

    backwards = ConstantMethod("backwards", 1, lambda s: -truth[s])
    result = evaluate_methods(
        [backwards],
        measurements,
        plan,
        campaign=with_threshold,
        k=5,
        n_bootstrap=50,
        seed=0,
    )

    assert result.primary_metric == "hit_rate_at_k"
    assert result.summaries["backwards"].mean("spearman") < 0
    assert result.recommendation.evidence_level == "exploratory"
    assert any("did not beat chance" in reason for reason in result.recommendation.reasons)


def test_chance_level_for_hit_rate_is_the_random_reference(campaign):
    measurements = _measurements_with_rounds(campaign, n=180, seed=4)
    threshold = float(measurements.frame["value"].quantile(0.8))
    with_threshold = _threshold_campaign(campaign, threshold)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))

    oracle = ConstantMethod("oracle", 1, lambda s: truth[s])
    result = evaluate_methods(
        [oracle], measurements, plan, campaign=with_threshold, k=5, n_bootstrap=50, seed=0
    )
    assert (
        result.summaries["oracle"].mean("hit_rate_at_k") > result.random_reference["hit_rate_at_k"]
    )
    assert result.recommendation.evidence_level == "supported"


def test_every_split_is_scored_at_the_same_budget(campaign):
    """Averaging metrics computed at different k would average different quantities."""
    import pandas as pd

    rng = np.random.default_rng(0)
    reference = campaign.reference_sequence
    positions = [3, 5, 7, 9, 11, 13, 15, 17]
    rows = []
    # Deliberately uneven rounds: 20, 10, then 40 measurements.
    for round_name, count in (("R1", 20), ("R2", 10), ("R3", 40)):
        for _ in range(count):
            residues = list(reference)
            for position in rng.choice(positions, size=2, replace=False):
                residues[position - 1] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
            rows.append(
                {
                    "variant_id": f"v{len(rows):03d}",
                    "sequence": "".join(residues),
                    "value": float(rng.normal()),
                    "condition_id": campaign.condition_id,
                    "round_id": round_name,
                }
            )
    measurements = load_measurements(pd.DataFrame(rows), campaign, strict=False)
    plan = choose_split_plan(measurements)
    assert len({s.n_test for s in plan.splits}) > 1, "the splits must differ in size"

    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    method = ConstantMethod("m", 1, lambda s: truth[s])
    result = evaluate_methods(
        [method], measurements, plan, campaign=campaign, k=24, n_bootstrap=20, seed=0
    )

    budgets = {m.k for m in result.summaries["m"].per_split.values()}
    assert budgets == {result.k}, f"splits scored at different budgets: {budgets}"


def test_the_resolution_is_the_half_width_of_the_interval():
    comparison = PairedComparison(
        method_key="b",
        reference_key="a",
        metric="spearman",
        difference=0.02,
        ci_low=-0.10,
        ci_high=0.14,
        fraction_better=0.6,
        n_bootstrap=500,
    )
    assert comparison.resolution == pytest.approx(0.12)
    assert not comparison.distinguishable


def test_a_null_result_says_what_it_could_have_detected(campaign):
    """'Not distinguishable' must distinguish 'equivalent' from 'we could not tell'."""
    measurements = _measurements_with_rounds(campaign, n=120, seed=7)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    rng = np.random.default_rng(0)

    simple = ConstantMethod("simple", 1, lambda s: truth[s] + rng.normal(0, 1.0))
    complex_ = ConstantMethod("complex", 3, lambda s: truth[s] + rng.normal(0, 1.0))
    result = evaluate_methods(
        [simple, complex_], measurements, plan, campaign=campaign, k=5, n_bootstrap=200, seed=0
    )

    recommendation = result.recommendation
    assert not recommendation.distinguishable
    assert recommendation.resolution is not None and recommendation.resolution > 0
    assert recommendation.n_test_sequences > 0

    statement = recommendation.power_statement
    assert f"{recommendation.n_test_sequences} distinct held-out sequences" in statement
    assert "could only resolve differences" in statement
    assert "not that the methods are equivalent" in statement
    assert recommendation.to_dict()["power_statement"] == statement


def test_an_adopted_improvement_is_reported_as_exceeding_the_resolution(campaign):
    measurements = _measurements_with_rounds(campaign, n=120, seed=8)
    plan = choose_split_plan(measurements)
    truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
    rng = np.random.default_rng(1)

    noise = ConstantMethod("simple", 1, lambda s: rng.normal())
    oracle = ConstantMethod("complex", 3, lambda s: truth[s])
    result = evaluate_methods(
        [noise, oracle], measurements, plan, campaign=campaign, k=5, n_bootstrap=200, seed=0
    )

    assert result.recommendation.distinguishable
    statement = result.recommendation.power_statement
    assert "the improvement it adopted exceeds that" in statement


def test_the_resolution_tightens_with_more_held_out_data(campaign):
    """More evidence must narrow what counts as undetectable, or the figure is not real."""
    truth_noise = 1.0

    def resolution_for(n: int) -> float:
        measurements = _measurements_with_rounds(campaign, n=n, seed=11)
        plan = choose_split_plan(measurements)
        truth = dict(zip(measurements.frame["sequence"], measurements.frame["value"], strict=True))
        rng = np.random.default_rng(3)
        methods = [
            ConstantMethod("simple", 1, lambda s: truth[s] + rng.normal(0, truth_noise)),
            ConstantMethod("complex", 3, lambda s: truth[s] + rng.normal(0, truth_noise)),
        ]
        result = evaluate_methods(
            methods, measurements, plan, campaign=campaign, k=5, n_bootstrap=300, seed=0
        )
        return result.recommendation.resolution

    small = resolution_for(60)
    large = resolution_for(240)
    assert small is not None and large is not None
    assert large < small, f"resolution did not tighten: {small:.3f} -> {large:.3f}"

"""Eligibility filters, selection policies, coverage, and rank variability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assaypilot import SelectionSettings, load_candidates, load_measurements
from assaypilot.data.validation import ExclusionCode
from assaypilot.evaluation.coverage import assess_coverage
from assaypilot.evaluation.uncertainty import (
    RANK_VARIABILITY_CAVEAT,
    bootstrap_rank_variability,
    interval_coverage,
)
from assaypilot.models.baseline import OneHotRidgeBaseline
from assaypilot.selection.filters import filter_candidates
from assaypilot.selection.policies import DiversityPolicy, TopRankedPolicy, build_policy


def _candidates(campaign, sequences, ids=None):
    frame = pd.DataFrame(
        {
            "candidate_id": ids or [f"c{i}" for i in range(1, len(sequences) + 1)],
            "sequence": sequences,
        }
    )
    return load_candidates(frame, campaign, strict=False)


# ------------------------------------------------------------------- filters


def test_locked_positions_are_enforced(campaign, reference, mutator):
    candidates = _candidates(campaign, [mutator(reference, (3, "G")), mutator(reference, (5, "W"))])
    settings = SelectionSettings(batch_size=2, locked_positions=(3,))
    result = filter_candidates(candidates, campaign, settings=settings)

    assert result.n_eligible == 1
    assert result.eligible.iloc[0]["candidate_id"] == "c2"
    assert result.counts_by_code() == {ExclusionCode.LOCKED_POSITION: 1}
    assert "locked position(s) 3" in result.exclusions[0].detail


def test_substitution_limit_is_enforced(campaign, reference, mutator):
    candidates = _candidates(
        campaign,
        [mutator(reference, (3, "G")), mutator(reference, (3, "G"), (5, "W"), (7, "D"))],
    )
    settings = SelectionSettings(batch_size=2, max_substitutions=2)
    result = filter_candidates(candidates, campaign, settings=settings)
    assert result.n_eligible == 1
    assert result.counts_by_code() == {ExclusionCode.TOO_MANY_SUBSTITUTIONS: 1}


def test_already_measured_candidates_are_excluded(campaign, reference, mutator):
    measured = mutator(reference, (3, "G"))
    measurements = load_measurements(
        pd.DataFrame(
            [{"variant_id": "v1", "sequence": measured, "value": 1.0, "condition_id": "37C"}]
        ),
        campaign,
    )
    candidates = _candidates(campaign, [measured, mutator(reference, (5, "W"))])

    result = filter_candidates(candidates, campaign, measurements=measurements)
    assert result.n_eligible == 1
    assert result.counts_by_code() == {ExclusionCode.ALREADY_MEASURED: 1}

    allowed = filter_candidates(
        candidates,
        campaign,
        measurements=measurements,
        settings=SelectionSettings(batch_size=2, allow_measured_repeats=True),
    )
    assert allowed.n_eligible == 2


def test_restriction_to_evaluated_range_is_recorded(campaign, reference, mutator):
    candidates = _candidates(campaign, [mutator(reference, (3, "G")), mutator(reference, (5, "W"))])
    result = filter_candidates(candidates, campaign, restrict_to_candidate_ids=["c1"])
    assert result.n_eligible == 1
    assert result.exclusions[0].code == "restricted"
    assert result.settings_applied["restricted_to_evaluated_range"] is True


def test_exclusions_table_has_stable_columns(campaign, reference, mutator):
    candidates = _candidates(campaign, [mutator(reference, (3, "G"))])
    result = filter_candidates(
        candidates, campaign, settings=SelectionSettings(batch_size=1, locked_positions=(3,))
    )
    table = result.exclusions_table()
    assert list(table.columns) == ["row", "identifier", "code", "severity", "detail"]


# ------------------------------------------------------------------ policies


def _ranked(campaign, reference, mutator):
    sequences = [
        mutator(reference, (3, "G")),
        mutator(reference, (3, "A")),  # one substitution away from the first
        mutator(reference, (11, "K"), (13, "E")),
        mutator(reference, (17, "P")),
    ]
    return pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "sequence": sequences,
            "score": [4.0, 3.9, 2.0, 1.0],
        }
    )


def test_top_ranked_policy_takes_the_best(campaign, reference, mutator):
    ranked = _ranked(campaign, reference, mutator)
    outcome = TopRankedPolicy().select(ranked, 2)
    assert list(outcome.selected["candidate_id"]) == ["c1", "c2"]
    assert outcome.shortfall == 0
    assert "rank 1 of 4" in outcome.selected.iloc[0]["selection_reason"]


def test_top_ranked_reports_a_short_pool(campaign, reference, mutator):
    ranked = _ranked(campaign, reference, mutator)
    outcome = TopRankedPolicy().select(ranked, 10)
    assert outcome.n_selected == 4
    assert outcome.shortfall == 6
    assert "only 4 eligible candidates" in outcome.shortfall_reason


def test_diversity_policy_suppresses_near_duplicates(campaign, reference, mutator):
    ranked = _ranked(campaign, reference, mutator)
    outcome = DiversityPolicy(min_distance=2).select(ranked, 2)
    # c2 differs from c1 at a single position, so it is suppressed.
    assert list(outcome.selected["candidate_id"]) == ["c1", "c3"]
    assert outcome.parameters["n_suppressed"] == 1
    assert "at least 2 substitutions" in outcome.selected.iloc[1]["selection_reason"]


def test_diversity_policy_reports_shortfall_without_relaxing(campaign, reference, mutator):
    ranked = _ranked(campaign, reference, mutator)
    outcome = DiversityPolicy(min_distance=40).select(ranked, 4)
    assert outcome.n_selected < 4
    assert outcome.shortfall > 0
    assert "was not relaxed" in outcome.shortfall_reason


def test_build_policy_validates_arguments():
    assert isinstance(build_policy("top_ranked"), TopRankedPolicy)
    assert isinstance(build_policy("diversity", min_distance=2), DiversityPolicy)
    with pytest.raises(ValueError, match="needs a minimum Hamming distance"):
        build_policy("diversity")
    with pytest.raises(ValueError, match="unknown selection policy"):
        build_policy("uncertainty")


# ------------------------------------------------------------------ coverage


def test_coverage_flags_candidates_beyond_the_evaluated_region(campaign, reference, mutator):
    measured = [mutator(reference, (3, "G")), mutator(reference, (5, "W"))]
    measurements = load_measurements(
        pd.DataFrame(
            [
                {"variant_id": f"v{i}", "sequence": s, "value": float(i), "condition_id": "37C"}
                for i, s in enumerate(measured)
            ]
        ),
        campaign,
    )
    near = mutator(reference, (3, "G"), (5, "W"))
    far = mutator(reference, (7, "D"), (9, "K"), (11, "R"), (13, "E"), (15, "Q"))
    candidates = _candidates(campaign, [near, far])

    report = assess_coverage(
        candidates,
        measurements,
        evaluated_sequences=[mutator(reference, (3, "G"), (5, "W"))],
        training_sequences=measured,
    )
    frame = report.frame.set_index("candidate_id")
    assert bool(frame.loc["c1", "within_evaluated_range"]) is True
    assert bool(frame.loc["c2", "within_evaluated_range"]) is False
    assert report.n_outside == 1
    assert any("further from the measured variants" in w for w in report.warnings)


def test_coverage_counts_unseen_substitutions(campaign, reference, mutator):
    measurements = load_measurements(
        pd.DataFrame(
            [
                {
                    "variant_id": "v1",
                    "sequence": mutator(reference, (3, "G")),
                    "value": 1.0,
                    "condition_id": "37C",
                }
            ]
        ),
        campaign,
    )
    candidates = _candidates(campaign, [mutator(reference, (3, "G"), (5, "W"))])
    report = assess_coverage(candidates, measurements)
    assert int(report.frame.iloc[0]["n_unseen_substitutions"]) == 1
    assert "never appears" in " ".join(report.warnings)


def test_coverage_without_evaluation_warns(campaign, reference, mutator):
    measurements = load_measurements(
        pd.DataFrame(
            [
                {
                    "variant_id": "v1",
                    "sequence": mutator(reference, (3, "G")),
                    "value": 1.0,
                    "condition_id": "37C",
                }
            ]
        ),
        campaign,
    )
    candidates = _candidates(campaign, [mutator(reference, (5, "W"))])
    report = assess_coverage(candidates, measurements)
    assert any("no candidate can be described as" in w for w in report.warnings)


# --------------------------------------------------------------- uncertainty


def test_rank_variability_reports_stability_not_probability(campaign, reference, mutator):
    rng = np.random.default_rng(0)
    positions = [3, 5, 7, 9, 11]
    sequences, values = [], []
    for _ in range(30):
        residues = list(reference)
        for position in rng.choice(positions, size=2, replace=False):
            residues[position - 1] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
        sequences.append("".join(residues))
        values.append(float(rng.normal()))

    candidates = [mutator(reference, (3, "G")), mutator(reference, (5, "W"))]
    variability = bootstrap_rank_variability(
        OneHotRidgeBaseline(reference, alphas=(1.0,)),
        train_sequences=sequences,
        y=np.array(values),
        groups=np.array(sequences),
        candidate_ids=["c1", "c2"],
        candidate_sequences=candidates,
        top_k=1,
        n_bootstrap=8,
        seed=0,
    )
    assert set(variability.frame.columns) == {
        "candidate_id",
        "mean_rank",
        "rank_std",
        "top_k_frequency",
    }
    assert variability.caveat == RANK_VARIABILITY_CAVEAT
    assert "not a calibrated probability" in variability.caveat
    assert (
        (variability.frame["top_k_frequency"] >= 0) & (variability.frame["top_k_frequency"] <= 1)
    ).all()


def test_interval_coverage_is_measurable():
    assert interval_coverage([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]) == pytest.approx(
        2 / 3
    )
    with pytest.raises(ValueError, match="same shape"):
        interval_coverage([1.0], [0.0, 0.0], [2.0, 2.0])

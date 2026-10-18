"""Import validation: what is kept, what is set aside, and why.

The rule under test throughout: nothing is dropped silently and nothing is
repaired silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from assaypilot import load_candidates, load_measurements
from assaypilot.data.loaders import apply_transform, candidates_from_sequences
from assaypilot.data.validation import ExclusionCode
from assaypilot.errors import InputError


def _frame(campaign, rows):
    return pd.DataFrame(rows)


def test_loads_valid_measurements(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    assert len(measurements) == 6
    assert measurements.n_unique_sequences == 5
    assert measurements.rounds() == ["R1", "R2"]
    assert measurements.report.n_rows_kept == 6
    assert not measurements.report.has_blocking


def test_replicates_share_a_group(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    groups = measurements.frame.loc[measurements.frame["variant_id"] == "var01", "seq_hash"]
    assert groups.nunique() == 1


def test_mutation_notation_is_accepted_and_checked(campaign, reference):
    rows = [
        {"variant_id": "a", "mutations": f"{reference[2]}3G", "value": 1.0, "condition_id": "37C"},
        {"variant_id": "b", "mutations": "WT", "value": 0.5, "condition_id": "37C"},
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    assert len(measurements) == 2
    assert measurements.frame.loc[0, "sequence"][2] == "G"
    assert measurements.frame.loc[1, "mutations"] == "WT"


def test_mutation_notation_mismatch_blocks_import(campaign, reference):
    wrong = "Z" if reference[2] != "Z" else "A"
    rows = [{"variant_id": "a", "mutations": f"{wrong}3G", "value": 1.0, "condition_id": "37C"}]
    with pytest.raises(InputError, match="cannot be used as given"):
        load_measurements(_frame(campaign, rows), campaign, strict=True)

    report = load_measurements(_frame(campaign, rows), campaign, strict=False).report
    assert report.counts_by_code() == {ExclusionCode.MUTATION_PARSE_ERROR: 1}
    assert report.has_blocking


def test_invalid_characters_are_blocking(campaign, reference):
    rows = [
        {"variant_id": "a", "sequence": "X" * len(reference), "value": 1.0, "condition_id": "37C"}
    ]
    report = load_measurements(_frame(campaign, rows), campaign, strict=False).report
    assert ExclusionCode.INVALID_CHARACTERS in report.counts_by_code()
    assert report.has_blocking


def test_length_mismatch_is_blocking(campaign, reference):
    rows = [{"variant_id": "a", "sequence": reference[:-2], "value": 1.0, "condition_id": "37C"}]
    report = load_measurements(_frame(campaign, rows), campaign, strict=False).report
    assert ExclusionCode.LENGTH_MISMATCH in report.counts_by_code()


def test_other_conditions_are_excluded_not_pooled(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 1.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 9.0,
            "condition_id": "50C",
        },
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign, strict=False)
    assert len(measurements) == 1
    codes = measurements.report.counts_by_code()
    assert codes[ExclusionCode.OTHER_CONDITION] == 1
    excluded = [
        e for e in measurements.report.exclusions if e.code == ExclusionCode.OTHER_CONDITION
    ]
    assert "separate campaign" in excluded[0].detail


def test_other_conditions_can_be_included_deliberately(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 1.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 9.0,
            "condition_id": "50C",
        },
    ]
    measurements = load_measurements(
        _frame(campaign, rows), campaign, strict=False, include_other_conditions=True
    )
    assert len(measurements) == 2


def test_censored_values_are_kept_but_not_fitted(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": "<0.01",
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": "2.0",
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign, strict=False)
    assert len(measurements) == 2
    assert set(measurements.frame["censoring"]) == {"left", "none"}
    for_fitting = measurements.for_regression()
    assert len(for_fitting) == 1
    assert any("censored" in note for note in for_fitting.report.notes)


def test_missing_and_non_numeric_values(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": "",
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": "high",
            "condition_id": "37C",
        },
    ]
    report = load_measurements(_frame(campaign, rows), campaign, strict=False).report
    codes = report.counts_by_code()
    assert codes[ExclusionCode.MISSING_VALUE] == 1
    assert codes[ExclusionCode.NON_NUMERIC_VALUE] == 1


def test_duplicate_identifier_with_different_sequence_is_blocking(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "same",
            "sequence": mutator(reference, (3, "G")),
            "value": 1.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "same",
            "sequence": mutator(reference, (5, "W")),
            "value": 2.0,
            "condition_id": "37C",
        },
    ]
    report = load_measurements(_frame(campaign, rows), campaign, strict=False).report
    assert report.counts_by_code()[ExclusionCode.DUPLICATE_ID] == 1


def test_missing_required_columns(campaign):
    with pytest.raises(InputError, match="needs a 'value' column"):
        load_measurements(pd.DataFrame({"variant_id": ["a"], "sequence": ["MK"]}), campaign)
    with pytest.raises(InputError, match="'sequence' column or a 'mutations' column"):
        load_measurements(pd.DataFrame({"variant_id": ["a"], "value": [1.0]}), campaign)


def test_conflicting_replicates_are_exposed(campaign, reference, mutator):
    sequence = mutator(reference, (3, "G"))
    rows = [
        {"variant_id": "a", "sequence": sequence, "value": 1.0, "condition_id": "37C"},
        {"variant_id": "a", "sequence": sequence, "value": 9.0, "condition_id": "37C"},
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    conflicts = measurements.conflicting_replicates()
    assert len(conflicts) == 1
    # Nothing was removed: both raw values are still present.
    assert len(measurements) == 2


def test_transform_keeps_raw_values(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 10.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 100.0,
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    transformed = apply_transform(measurements, "log10")
    assert list(transformed.frame["value"]) == [1.0, 2.0]
    assert list(transformed.frame["value_raw"]) == [10.0, 100.0]
    assert transformed.transform == "log10"


def test_log_transform_refuses_non_positive_values(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 0.0,
            "condition_id": "37C",
        }
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    with pytest.raises(InputError, match="zero or negative"):
        apply_transform(measurements, "log10")


def test_duplicate_candidates_are_excluded(campaign, reference, mutator):
    sequence = mutator(reference, (3, "G"))
    frame = pd.DataFrame({"candidate_id": ["c1", "c2"], "sequence": [sequence, sequence]})
    candidates = load_candidates(frame, campaign, strict=False)
    assert len(candidates) == 1
    assert candidates.report.counts_by_code()[ExclusionCode.DUPLICATE_SEQUENCE] == 1


def test_candidates_from_fasta(tmp_path, campaign, reference, mutator):
    path = tmp_path / "candidates.fasta"
    path.write_text(
        f">c1 first\n{mutator(reference, (3, 'G'))}\n>c2\n{mutator(reference, (5, 'W'))}\n",
        encoding="utf-8",
    )
    candidates = load_candidates(path, campaign)
    assert candidates.ids == ["c1", "c2"]


def test_candidates_from_sequences_helper(campaign, reference, mutator):
    candidates = candidates_from_sequences(
        [mutator(reference, (3, "G")), mutator(reference, (5, "W"))], campaign
    )
    assert len(candidates) == 2
    assert candidates.ids == ["cand00001", "cand00002"]


def test_unsupported_file_type(tmp_path, campaign):
    path = tmp_path / "data.xyz"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(InputError, match="unsupported file type"):
        load_measurements(path, campaign)


def test_report_summary_lines_mention_counts(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    lines = measurements.report.summary_lines()
    assert "6 of 6 rows usable" in lines[0]


# ------------------------------------- transforms, thresholds, and censoring


def _log_scale_campaign(campaign, threshold: float):
    return campaign.__class__.from_dict(
        {**campaign.to_dict(), "assay_units": "RFU", "success_threshold": threshold}
    )


def test_transform_maps_the_success_threshold_into_value_space(campaign, reference, mutator):
    """A log-transformed value must not be compared against a raw threshold."""
    with_threshold = _log_scale_campaign(campaign, 1000.0)
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 100.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 10000.0,
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(with_threshold, rows), with_threshold)

    assert measurements.threshold_in_value_space() == pytest.approx(1000.0)
    transformed = apply_transform(measurements, "log10")
    assert transformed.threshold_in_value_space() == pytest.approx(3.0)

    # The hit flags are the same variants before and after the transform.
    assert list(measurements.is_hit_array(measurements.values)) == [False, True]
    assert list(transformed.is_hit_array(transformed.values)) == [False, True]


def test_transform_renames_the_value_units(campaign, reference, mutator):
    with_threshold = _log_scale_campaign(campaign, 1000.0)
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 100.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 10000.0,
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(with_threshold, rows), with_threshold)
    assert measurements.value_units() == "RFU"
    assert apply_transform(measurements, "log10").value_units() == "log10(RFU)"
    assert apply_transform(measurements, "zscore").value_units() == "z-score of RFU"


def test_a_threshold_a_transform_cannot_express_yields_no_hits(campaign, reference, mutator):
    """A non-positive threshold under a log transform disables hit rates, not lies."""
    with_threshold = _log_scale_campaign(campaign, -5.0)
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": 100.0,
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": 10000.0,
            "condition_id": "37C",
        },
    ]
    measurements = apply_transform(
        load_measurements(_frame(with_threshold, rows), with_threshold), "log10"
    )
    assert measurements.threshold_in_value_space() is None
    assert measurements.is_hit_array(measurements.values) is None


def test_zscore_threshold_follows_the_same_scaling(campaign, reference, mutator):
    with_threshold = _log_scale_campaign(campaign, 3.0)
    rows = [
        {
            "variant_id": f"v{i}",
            "sequence": mutator(reference, (3 + i, "G")),
            "value": float(i),
            "condition_id": "37C",
        }
        for i in range(1, 6)
    ]
    measurements = load_measurements(_frame(with_threshold, rows), with_threshold)
    transformed = apply_transform(measurements, "zscore")

    raw_hits = list(measurements.is_hit_array(measurements.values))
    transformed_hits = list(transformed.is_hit_array(transformed.values))
    assert raw_hits == transformed_hits


def test_for_regression_records_every_dropped_row(campaign, reference, mutator):
    """Rows removed before fitting must appear in the exclusions, not only in a note."""
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": "<0.05",
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": "2.0",
            "condition_id": "37C",
        },
        {
            "variant_id": "c",
            "sequence": mutator(reference, (7, "D")),
            "value": ">9.0",
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    fitting = measurements.for_regression()

    assert len(fitting) == 1
    codes = fitting.report.counts_by_code()
    assert codes[ExclusionCode.CENSORED_VALUE] == 2

    censored = [e for e in fitting.report.exclusions if e.code == ExclusionCode.CENSORED_VALUE]
    assert {e.identifier for e in censored} == {"a", "c"}
    assert all(e.row is not None for e in censored)
    assert "below a limit" in censored[0].detail


def test_for_regression_is_idempotent(campaign, reference, mutator):
    rows = [
        {
            "variant_id": "a",
            "sequence": mutator(reference, (3, "G")),
            "value": "<0.05",
            "condition_id": "37C",
        },
        {
            "variant_id": "b",
            "sequence": mutator(reference, (5, "W")),
            "value": "2.0",
            "condition_id": "37C",
        },
    ]
    measurements = load_measurements(_frame(campaign, rows), campaign)
    once = measurements.for_regression()
    twice = once.for_regression()
    assert len(twice) == len(once)
    assert twice.report.counts_by_code() == once.report.counts_by_code()

"""End-to-end planning: the plan, its labels, its determinism, its exports."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from assaypilot import SelectionSettings, load_candidates, plan_batch
from assaypilot.errors import InsufficientDataError
from assaypilot.pipeline import design_id
from assaypilot.reporting.exports import budget_frame, export_plan, method_comparison_frame
from assaypilot.reporting.html import render_report


def _plan(simulated_inputs, physchem_suite, **kwargs):
    campaign, measurements, candidates = simulated_inputs
    options = {
        "suite": physchem_suite,
        "n_bootstrap": 50,
        "seed": 0,
    }
    options.update(kwargs)
    return plan_batch(campaign, candidates, measurements, **options)


def test_plan_selects_a_full_batch(simulated_inputs, physchem_suite):
    campaign, _, _ = simulated_inputs
    plan = _plan(simulated_inputs, physchem_suite)

    assert len(plan.selected) == campaign.selection.batch_size
    assert plan.outcome.shortfall == 0
    assert plan.evidence_level in {"supported", "exploratory"}
    assert plan.evaluation is not None
    assert plan.method_key in plan.evaluation.summaries


def test_selected_rows_carry_their_provenance(simulated_inputs, physchem_suite):
    campaign, _, _ = simulated_inputs
    plan = _plan(simulated_inputs, physchem_suite)
    row = plan.selected.iloc[0]

    assert row["design_id"] == design_id(campaign, row["sequence"])
    assert row["run_id"] == plan.run_id
    assert row["campaign_id"] == campaign.campaign_id
    assert row["method"] == plan.method_key
    assert row["evidence_level"] == plan.evidence_level
    assert row["selection_reason"]


def test_design_id_is_stable_across_runs(simulated_inputs, physchem_suite):
    first = _plan(simulated_inputs, physchem_suite)
    second = _plan(simulated_inputs, physchem_suite)
    assert list(first.selected["design_id"]) == list(second.selected["design_id"])


def test_plan_is_deterministic(simulated_inputs, physchem_suite):
    first = _plan(simulated_inputs, physchem_suite)
    second = _plan(simulated_inputs, physchem_suite)

    assert list(first.selected["candidate_id"]) == list(second.selected["candidate_id"])
    np.testing.assert_allclose(first.selected["score"], second.selected["score"])
    assert first.manifest.fingerprint() == second.manifest.fingerprint()


def test_different_seeds_do_not_change_a_deterministic_ranking(simulated_inputs, physchem_suite):
    """Seeds affect resampling, not the ranking of a deterministic method."""
    first = _plan(simulated_inputs, physchem_suite, seed=0)
    second = _plan(simulated_inputs, physchem_suite, seed=5)
    assert list(first.ranking["candidate_id"]) == list(second.ranking["candidate_id"])


def test_predicted_values_are_separate_from_observations(simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite)
    assert "predicted_value" in plan.ranking
    assert "observed_value" in plan.ranking
    # No candidate in the pool has been measured, so observations stay empty.
    assert plan.ranking["observed_value"].isna().all()


def test_diversity_policy_changes_the_batch(simulated_inputs, physchem_suite):
    campaign, measurements, candidates = simulated_inputs
    settings = SelectionSettings(batch_size=8, diversity_min_distance=4)
    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        settings=settings,
        suite=physchem_suite,
        policy="diversity",
        n_bootstrap=20,
    )
    sequences = plan.selected["sequence"].tolist()
    for i, a in enumerate(sequences):
        for b in sequences[i + 1 :]:
            assert sum(1 for x, y in zip(a, b, strict=True) if x != y) >= 4


def test_locked_positions_never_appear_in_the_batch(simulated_inputs, physchem_suite):
    campaign, measurements, candidates = simulated_inputs
    locked = int(candidates.frame["mutations"].iloc[0].split(":")[0][1:-1])
    settings = SelectionSettings(batch_size=6, locked_positions=(locked,))
    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        settings=settings,
        suite=physchem_suite,
        n_bootstrap=20,
    )
    for mutations in plan.selected["mutations"]:
        positions = [int(m[1:-1]) for m in str(mutations).split(":") if mutations != "WT"]
        assert locked not in positions


def test_restricting_to_the_evaluated_range_excludes_far_candidates(
    simulated_inputs, physchem_suite
):
    campaign, measurements, candidates = simulated_inputs
    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        suite=physchem_suite,
        restrict_to_evaluated_range=True,
        n_bootstrap=20,
    )
    assert plan.selected["within_evaluated_range"].all()
    assert any("restricted" in note for note in plan.notes)


def test_budget_separates_candidates_from_controls(simulated_inputs, physchem_suite):
    campaign, measurements, candidates = simulated_inputs
    settings = SelectionSettings(
        batch_size=6, replicates_per_candidate=2, control_sequences=("WT", "empty")
    )
    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        settings=settings,
        suite=physchem_suite,
        n_bootstrap=20,
    )
    budget = plan.budget
    assert budget["candidate_wells"] == 12
    assert budget["control_wells"] == 2
    assert budget["total_wells"] == 14
    table = budget_frame(plan)
    assert set(table["item"]) >= {"candidate wells", "control wells", "total wells"}


def test_plan_without_measurements_is_labelled_unvalidated(simulated_inputs, physchem_suite):
    """With no labels and no label-free method, planning fails with a reason
    rather than producing something that looks validated. The complementary
    case — a label-free ranking, labelled unvalidated — is in tests/test_esm.py."""
    from assaypilot.features.esm import esm_available

    del esm_available
    campaign, _, candidates = simulated_inputs

    # This suite has no label-free method, so there is nothing that could rank
    # candidates without measurements, and the refusal must say so.
    with pytest.raises(InsufficientDataError, match="No measured data"):
        plan_batch(campaign, candidates, None, suite=physchem_suite)


def test_empty_candidate_pool_is_rejected(simulated_inputs, physchem_suite):
    campaign, measurements, _ = simulated_inputs
    empty = load_candidates(
        pd.DataFrame(columns=["candidate_id", "sequence"]), campaign, strict=False
    )
    with pytest.raises(InsufficientDataError, match="candidate pool is empty"):
        plan_batch(campaign, empty, measurements, suite=physchem_suite)


def test_a_method_that_cannot_run_is_named_rather_than_skipped(simulated_inputs, physchem_suite):
    """Whether or not ESM is installed, the report must account for all three methods."""
    from dataclasses import replace as replace_dataclass

    from assaypilot.features.esm import esm_available

    if esm_available():
        pytest.skip("ESM is installed; the available case is covered by tests/test_esm.py")

    campaign, measurements, candidates = simulated_inputs
    with_pretrained = replace_dataclass(physchem_suite, include_pretrained=True)
    plan = plan_batch(
        campaign, candidates, measurements, suite=with_pretrained, n_bootstrap=20, seed=0
    )

    assert plan.evaluation is not None
    assert "pretrained_esm_masked_marginal" in plan.evaluation.unavailable
    reason = plan.evaluation.unavailable["pretrained_esm_masked_marginal"]
    assert "torch" in reason and "assaypilot[esm]" in reason
    assert any("pretrained" in warning for warning in plan.warnings)


def test_rank_variability_is_attached_when_requested(simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite, rank_variability_refits=5)
    assert plan.rank_variability is not None
    assert "top_k_frequency" in plan.selected.columns
    assert plan.rank_variability.n_bootstrap == 5


def test_manifest_records_what_reproduction_needs(simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite)
    manifest = plan.manifest.to_dict()

    assert manifest["config"]["seed"] == 0
    assert manifest["config"]["suite"]["representation"] == "physchem"
    assert manifest["config"]["method_used"]["key"] == plan.method_key
    assert manifest["environment"]["packages"]["numpy"] != "not installed"
    assert manifest["evaluation"]["partitions"], "partition assignments must be recorded"
    assert len(manifest["predictions"]) == len(plan.candidates)
    assert len(manifest["selected"]) == len(plan.selected)
    assert manifest["campaign"]["reference_sequence"]


def test_manifest_round_trips_through_json(tmp_path, simulated_inputs, physchem_suite):
    from assaypilot.reporting.manifest import RunManifest

    plan = _plan(simulated_inputs, physchem_suite)
    path = plan.manifest.write(tmp_path / "manifest.json")
    restored = RunManifest.read(path)
    assert restored.run_id == plan.run_id
    assert restored.fingerprint() == plan.manifest.fingerprint()
    json.loads(path.read_text(encoding="utf-8"))


def test_exports_write_every_artifact(tmp_path, simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite, rank_variability_refits=3)
    written = export_plan(plan, tmp_path)

    for key in (
        "selected_variants",
        "selected_fasta",
        "all_predictions",
        "exclusions",
        "manifest",
        "campaign",
        "coverage",
        "method_comparison",
        "report",
    ):
        assert written[key].exists(), key
        assert written[key].stat().st_size > 0

    selected = pd.read_csv(written["selected_variants"])
    assert list(selected.columns)[:4] == [
        "design_id",
        "candidate_id",
        "selection_rank",
        "sequence",
    ]
    assert len(selected) == len(plan.selected)

    fasta = written["selected_fasta"].read_text(encoding="utf-8")
    assert fasta.startswith(">")
    assert plan.selected.iloc[0]["design_id"] in fasta

    comparison = method_comparison_frame(plan)
    assert set(comparison["scope"]) == {"split", "aggregate"}


def test_report_states_the_evidence_level(simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite)
    html = render_report(plan)

    assert "<!doctype html>" in html.lower()
    assert plan.campaign.assay_name in html
    assert plan.run_id in html
    assert ("Supported by held-out evaluation" in html) or ("Exploratory" in html)
    assert "Reproducing this run" in html
    # The selection rule is quoted verbatim in the report.
    assert "simplest available method" in html


def test_headline_names_method_policy_and_evidence(simulated_inputs, physchem_suite):
    plan = _plan(simulated_inputs, physchem_suite)
    headline = plan.headline()
    assert plan.method_name in headline
    assert plan.outcome.policy in headline
    assert "candidates selected" in headline


def test_censored_measurements_never_reach_the_model(simulated_inputs, physchem_suite):
    """plan_batch excludes them itself, so every caller is protected, not just the CLI."""
    import pandas as pd

    campaign, measurements, candidates = simulated_inputs
    frame = measurements.frame.copy()
    censored_ids = frame["variant_id"].tolist()[:10]
    raw = pd.DataFrame(
        {
            "variant_id": frame["variant_id"],
            "sequence": frame["sequence"],
            "value": [
                f"<{abs(v):.3f}" if vid in censored_ids else f"{v}"
                for vid, v in zip(frame["variant_id"], frame["value"], strict=True)
            ],
            "condition_id": frame["condition_id"],
            "round_id": frame["round_id"],
        }
    )
    from assaypilot import load_measurements

    with_censored = load_measurements(raw, campaign, strict=False)
    assert int((with_censored.frame["censoring"] != "none").sum()) > 0

    plan = plan_batch(
        campaign, candidates, with_censored, suite=physchem_suite, n_bootstrap=20, seed=0
    )

    # Nothing censored survived into the data the model saw.
    assert plan.measurements is not None
    assert (plan.measurements.frame["censoring"] == "none").all()

    # And every removed row is in the exported record, not just a note.
    from assaypilot.data.validation import ExclusionCode

    # One exclusion per row, not per variant: replicates are recorded separately.
    codes = [e["code"] for e in plan.manifest.exclusions]
    assert codes.count(ExclusionCode.CENSORED_VALUE) == len(censored_ids)


def test_predicted_values_are_labelled_with_the_transformed_units(simulated_inputs, physchem_suite):
    """A prediction fitted on log10 values is not a number of raw assay units."""
    from assaypilot.data.loaders import apply_transform

    campaign, measurements, candidates = simulated_inputs
    positive = measurements.frame.copy()
    positive["value"] = positive["value"] - positive["value"].min() + 1.0
    positive["value_raw"] = positive["value"]
    shifted = type(measurements)(campaign=campaign, frame=positive, report=measurements.report)
    transformed = apply_transform(shifted, "log10")

    plan = plan_batch(
        campaign, candidates, transformed, suite=physchem_suite, n_bootstrap=20, seed=0
    )
    units = set(plan.selected["predicted_value_units"])
    assert units == {f"log10({campaign.assay_units})"}


def test_coverage_is_measured_against_each_split_s_own_training_set(
    simulated_inputs, physchem_suite
):
    """Pooling training rows across splits would make held-out distances come out as zero."""
    plan = _plan(simulated_inputs, physchem_suite)
    summary = plan.coverage.summary

    assert plan.evaluation is not None and len(plan.evaluation.split_plan.splits) > 1
    assert summary["max_evaluated_distance"] is not None
    assert summary["evaluated_distance_median"] is not None
    # A sequence held out in one split is training data in another; if the
    # pooled union were used, these would collapse towards zero.
    assert summary["evaluated_distance_median"] > 0
    assert summary["max_evaluated_distance"] > 0


def test_unseen_substitutions_are_counted_against_all_measurements(
    simulated_inputs, physchem_suite
):
    """The warning says 'never appears in the measurements', so that is what is counted."""
    from assaypilot.sequences import substitutions

    campaign, measurements, _ = simulated_inputs
    plan = _plan(simulated_inputs, physchem_suite)

    measured = {
        (s.position, s.mutant)
        for sequence in measurements.frame["sequence"]
        for s in substitutions(campaign.reference_sequence, sequence)
    }
    for row in plan.coverage.frame.itertuples(index=False):
        candidate = plan.candidates.frame.set_index("candidate_id").loc[row.candidate_id]
        expected = sum(
            1
            for s in substitutions(campaign.reference_sequence, candidate["sequence"])
            if (s.position, s.mutant) not in measured
        )
        assert int(row.n_unseen_substitutions) == expected

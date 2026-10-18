"""Round history: append-only storage, and scoring a plan against its results.

The behaviours under test are the ones that make a campaign auditable months
later: a new round never edits an old plan, and results are matched back to
predictions by an identifier that cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assaypilot import CampaignStore, load_measurements, plan_batch, score_round
from assaypilot.errors import CampaignError
from assaypilot.pipeline import design_id


@pytest.fixture
def planned(simulated_inputs, physchem_suite):
    campaign, measurements, candidates = simulated_inputs
    plan = plan_batch(
        campaign, candidates, measurements, suite=physchem_suite, n_bootstrap=20, seed=0
    )
    return campaign, measurements, plan


def _results_for(plan, campaign, *, noise=0.2, seed=0, fraction=1.0):
    """Fabricate measurements for a planned batch, as if the plate came back."""
    rng = np.random.default_rng(seed)
    selected = plan.selected
    take = max(1, int(len(selected) * fraction))
    rows = []
    for _i, row in enumerate(selected.head(take).itertuples(index=False)):
        rows.append(
            {
                "variant_id": row.design_id,
                "sequence": row.sequence,
                "value": float(row.score + rng.normal(0, noise)),
                "condition_id": campaign.condition_id,
                "round_id": "R4",
            }
        )
    return load_measurements(pd.DataFrame(rows), campaign, strict=True)


# ------------------------------------------------------------------- scoring


def test_scorecard_matches_results_by_design_id(planned):
    campaign, measurements, plan = planned
    results = _results_for(plan, campaign)

    card = score_round(
        plan.manifest, results, campaign=campaign, previous_measurements=measurements
    )

    assert card.run_id == plan.run_id
    assert card.n_planned == len(plan.selected)
    assert card.n_matched == len(plan.selected)
    assert card.coverage == pytest.approx(1.0)
    assert np.isfinite(card.metrics["spearman"])
    assert "previous_best" in card.metrics
    assert any("does not say what a different selection" in note for note in card.notes)


def test_scorecard_reports_partial_returns(planned):
    campaign, _, plan = planned
    results = _results_for(plan, campaign, fraction=0.5)
    card = score_round(plan.manifest, results, campaign=campaign)

    assert card.n_matched < card.n_planned
    assert card.coverage < 1.0
    assert any("no measured result" in note for note in card.notes)


def test_scorecard_without_a_threshold_says_so(simulated_inputs, physchem_suite):
    campaign, measurements, candidates = simulated_inputs
    no_threshold = campaign.__class__.from_dict({**campaign.to_dict(), "success_threshold": None})
    plan = plan_batch(no_threshold, candidates, measurements, suite=physchem_suite, n_bootstrap=20)
    results = _results_for(plan, no_threshold)
    card = score_round(plan.manifest, results, campaign=no_threshold)
    assert "hit_rate" not in card.metrics
    assert any("No success threshold" in note for note in card.notes)


def test_scorecard_summary_is_human_readable(planned):
    campaign, measurements, plan = planned
    card = score_round(plan.manifest, _results_for(plan, campaign), campaign=campaign)
    text = "\n".join(card.summary_lines())
    assert plan.run_id in text
    assert "Rank correlation" in text


def test_design_ids_are_stable_so_results_can_be_matched(planned):
    campaign, _, plan = planned
    for row in plan.selected.itertuples(index=False):
        assert row.design_id == design_id(campaign, row.sequence)


# --------------------------------------------------------------------- store


def test_store_round_trip(tmp_path, planned):
    campaign, measurements, plan = planned
    with CampaignStore(tmp_path / "campaigns.db") as store:
        version = store.create_campaign(campaign)
        assert version == 1
        store.add_round(campaign.campaign_id, measurements, round_id="R1")
        store.save_run(plan)

        restored = store.get_campaign(campaign.campaign_id)
        assert restored.reference_sequence == campaign.reference_sequence

        runs = store.list_runs(campaign.campaign_id)
        assert list(runs["run_id"]) == [plan.run_id]
        assert runs.iloc[0]["n_selected"] == len(plan.selected)

        selections = store.selections(plan.run_id)
        assert len(selections) == len(plan.selected)
        assert list(selections["selection_rank"]) == sorted(selections["selection_rank"])

        stored_run = store.get_run(plan.run_id)
        assert stored_run.manifest.run_id == plan.run_id
        assert stored_run.evidence_level == plan.evidence_level


def test_adding_a_round_creates_a_new_version_without_touching_history(tmp_path, planned):
    campaign, measurements, plan = planned
    results = _results_for(plan, campaign)

    with CampaignStore(tmp_path / "campaigns.db") as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")
        store.save_run(plan)
        before = store.selections(plan.run_id)

        new_version = store.add_round(campaign.campaign_id, results, round_id="R4")
        after = store.selections(plan.run_id)

        assert new_version == 3  # created, first round, second round
        pd.testing.assert_frame_equal(before, after)

        versions = store.campaign_versions(campaign.campaign_id)
        assert list(versions["version"]) == [1, 2, 3]
        assert "R4" in versions.iloc[-1]["note"]

        stored = store.load_measurements(campaign.campaign_id)
        assert len(stored) == len(measurements) + len(results)
        assert "R4" in [str(r) for r in stored.rounds()]


def test_history_joins_runs_with_their_scorecards(tmp_path, planned):
    campaign, measurements, plan = planned
    results = _results_for(plan, campaign)

    with CampaignStore(tmp_path / "campaigns.db") as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")
        store.save_run(plan)
        card = score_round(plan.manifest, results, campaign=campaign)
        store.save_scorecard(card)

        history = store.history(campaign.campaign_id)
        assert len(history) == 1
        assert history.iloc[0]["n_matched"] == card.n_matched
        assert history.iloc[0]["run_id"] == plan.run_id


def test_duplicate_campaign_is_rejected(tmp_path, planned):
    campaign, _, _ = planned
    with CampaignStore(tmp_path / "campaigns.db") as store:
        store.create_campaign(campaign)
        with pytest.raises(CampaignError, match="already exists"):
            store.create_campaign(campaign)


def test_missing_campaign_and_run_raise(tmp_path):
    with CampaignStore(tmp_path / "campaigns.db") as store:
        with pytest.raises(CampaignError, match="no campaign"):
            store.get_campaign("nope")
        with pytest.raises(CampaignError, match="no run"):
            store.get_run("nope")


def test_list_campaigns_counts_rounds_and_runs(tmp_path, planned):
    campaign, measurements, plan = planned
    with CampaignStore(tmp_path / "campaigns.db") as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")
        store.save_run(plan)
        listing = store.list_campaigns()
        row = listing.iloc[0]
        assert row["campaign_id"] == campaign.campaign_id
        assert row["n_runs"] == 1
        assert row["n_measurements"] == len(measurements)

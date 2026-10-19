"""The interface, driven headlessly.

Two things are checked here. First that the app runs at all — every screen
executes on load, so a typo in a rarely-visited tab cannot ship unnoticed.
Second, and the reason the app was wired to the campaign store: that a plan
survives the browser tab. The round loop the README leads with is only real if
a plan made in one session can be scored in another, and that is what
:func:`test_a_plan_survives_a_new_session` demonstrates.

Skipped when Streamlit is not installed, so the core suite stays dependency-free.
"""

from __future__ import annotations

import pandas as pd
import pytest

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="the app extra is not installed"
)
AppTest = streamlit_testing.AppTest

from assaypilot import (  # noqa: E402 - after the optional-dependency guard
    CampaignStore,
    SelectionSettings,
    load_measurements,
    plan_batch,
    score_round,
)
from assaypilot.app import streamlit_app  # noqa: E402

APP_PATH = streamlit_app.__file__


def _run_app(store_path, timeout: float = 60):
    app = AppTest.from_file(APP_PATH, default_timeout=timeout)
    app.session_state["store_path"] = str(store_path)
    return app.run()


def test_every_screen_runs_on_a_cold_start(tmp_path):
    app = _run_app(tmp_path / "campaigns.db")
    assert not app.exception
    headings = " ".join(h.value for h in app.header)
    for screen in ("Campaign", "Data review", "Evidence and batch", "Next round"):
        assert screen in headings


def test_the_store_path_is_configurable_and_local(tmp_path):
    app = _run_app(tmp_path / "elsewhere.db")
    assert not app.exception
    assert app.session_state["store_path"].endswith("elsewhere.db")


def test_an_empty_store_offers_no_saved_campaigns(tmp_path):
    app = _run_app(tmp_path / "campaigns.db")
    assert not app.exception
    assert not any("Open a saved campaign" in s.value for s in app.subheader)


def test_a_saved_campaign_is_offered_when_one_exists(tmp_path, simulated_inputs):
    campaign, measurements, _ = simulated_inputs
    store_path = tmp_path / "campaigns.db"
    with CampaignStore(store_path) as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")

    app = _run_app(store_path)
    assert not app.exception
    assert any("Open a saved campaign" in s.value for s in app.subheader)
    labels = app.selectbox(key="saved_pick").options
    assert any(campaign.campaign_id in label for label in labels)


def test_a_plan_survives_a_new_session(tmp_path, simulated_inputs, physchem_suite):
    """The round loop across sessions: plan now, score weeks later.

    This is the behaviour the interface previously could not support, because
    it held everything in memory. The store is what the app reads, so this
    exercises the same path the screens take.
    """
    campaign, measurements, candidates = simulated_inputs
    store_path = tmp_path / "campaigns.db"

    # --- session one: a campaign, a round of data, a plan
    with CampaignStore(store_path) as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")

    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        settings=SelectionSettings(batch_size=6),
        suite=physchem_suite,
        n_bootstrap=20,
    )
    with CampaignStore(store_path) as store:
        store.save_run(plan)

    # --- the browser tab closes; nothing is carried over but the file path
    del plan, measurements

    app = _run_app(store_path)
    assert not app.exception

    # --- session two: the plan is still there, with its selections
    with CampaignStore(store_path) as store:
        runs = store.list_runs(campaign.campaign_id)
        assert len(runs) == 1
        run_id = runs.iloc[0]["run_id"]
        restored = store.get_run(run_id)
        selections = store.selections(run_id)
        stored_measurements = store.load_measurements(campaign.campaign_id)

    assert len(selections) == 6
    assert restored.manifest.run_id == run_id

    # --- the plate reads out, and the old predictions are scored
    results = pd.DataFrame(
        {
            "variant_id": selections["design_id"],
            "sequence": selections["sequence"],
            "value": selections["score"] * 0.9 + 0.2,
            "condition_id": campaign.condition_id,
            "round_id": "R2",
        }
    )
    new_round = load_measurements(results, campaign, strict=True)
    scorecard = score_round(
        restored.manifest,
        new_round,
        campaign=campaign,
        previous_measurements=stored_measurements,
    )
    assert scorecard.n_matched == 6
    assert scorecard.run_id == run_id

    # --- and the round is appended without disturbing what came before
    with CampaignStore(store_path) as store:
        before = store.selections(run_id)
        store.save_scorecard(scorecard)
        store.add_round(campaign.campaign_id, new_round, round_id="R2")
        after = store.selections(run_id)
        history = store.history(campaign.campaign_id)
        rounds_now = store.load_measurements(campaign.campaign_id)

    pd.testing.assert_frame_equal(before, after)
    assert history.iloc[0]["n_matched"] == 6
    assert "R2" in {str(r) for r in rounds_now.rounds()}


def test_the_app_reads_the_same_history_the_cli_writes(tmp_path, simulated_inputs):
    """One store, two front ends: a campaign made at the command line opens here."""
    campaign, measurements, _ = simulated_inputs
    store_path = tmp_path / "campaigns.db"
    with CampaignStore(store_path) as store:
        store.create_campaign(campaign)
        store.add_round(campaign.campaign_id, measurements, round_id="R1")

    app = _run_app(store_path)
    assert not app.exception
    assert any(campaign.campaign_id in label for label in app.selectbox(key="saved_pick").options)


def test_a_missing_store_file_is_not_an_error(tmp_path):
    """Pointing at a path that does not exist yet must not break the first screen."""
    app = _run_app(tmp_path / "nested" / "does-not-exist-yet.db")
    assert not app.exception

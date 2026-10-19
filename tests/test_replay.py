"""The retrospective replay harness.

What is checked: every strategy gets the same starting point, budgets are
respected, nothing is selected twice, and the limits of the exercise travel
with the result.
"""

from __future__ import annotations

import pytest

from assaypilot.benchmarks.replay import (
    REPLAY_CAVEAT,
    ReplayConfig,
    compare_strategies,
    replay_campaign,
    strategy_names,
)
from assaypilot.models.registry import SuiteConfig


@pytest.fixture
def replay_config(tmp_path) -> ReplayConfig:
    return ReplayConfig(
        n_initial=30,
        batch_size=6,
        n_rounds=2,
        seeds=(0, 1),
        diversity_min_distance=2,
        # No pretrained method: the fast suite must not download a checkpoint
        # just because torch happens to be installed.
        suite=SuiteConfig(
            representation="physchem",
            cache_dir=str(tmp_path / "cache"),
            physchem_bins=4,
            include_pretrained=False,
        ),
    )


def test_replay_runs_every_strategy(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    expected = set(strategy_names(replay_config, simulated.campaign))

    assert set(result.frame["strategy"]) == expected
    assert "random" in expected
    assert set(result.frame["round"]) == {1, 2}
    assert set(result.frame["seed"]) == {0, 1}


def test_replay_respects_the_budget(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    assert (result.frame["batch_size"] <= replay_config.batch_size).all()

    for (_strategy, _seed), group in result.frame.groupby(["strategy", "seed"]):
        rounds = group.sort_values("round")
        experiments = rounds["cumulative_experiments"].tolist()
        # Strictly increasing: nothing is ever "re-measured" in a replay.
        assert experiments == sorted(set(experiments))
        assert experiments[0] <= replay_config.n_initial + replay_config.batch_size


def test_best_found_never_decreases(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    for _, group in result.frame.groupby(["strategy", "seed"]):
        values = group.sort_values("round")["best_found"].tolist()
        assert values == sorted(values)


def test_replay_is_reproducible(simulated, replay_config):
    first = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    second = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    columns = ["strategy", "seed", "round", "best_found", "cumulative_experiments"]
    assert first.frame[columns].equals(second.frame[columns])


def test_summary_and_comparison_tables(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)

    summary = result.summary()
    assert {
        "strategy",
        "best_selected_mean",
        "best_found_mean",
        "experiments",
    } <= set(summary.columns)
    assert len(summary) == len(set(result.frame["strategy"]))

    comparison = compare_strategies(result, reference="random")
    assert "random" not in set(comparison["strategy"])
    assert {"mean_difference", "n_seeds", "wins", "metric"} <= set(comparison.columns)
    assert set(comparison["metric"]) == {"best_selected"}
    assert (comparison["n_seeds"] == len(replay_config.seeds)).all()


def test_caveat_travels_with_the_result(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    from assaypilot.features.esm import esm_available

    assert REPLAY_CAVEAT in result.caveat
    assert "Retrospective replay" in result.to_dict()["caveat"]
    # This replay excludes the pretrained method outright, so nothing should be
    # reported as unavailable whether or not torch is installed.
    del esm_available
    assert "not available in this environment" not in result.caveat


def test_replay_refuses_an_undersized_pool(simulated):
    config = ReplayConfig(n_initial=10_000, batch_size=24, n_rounds=1, seeds=(0,))
    with pytest.raises(ValueError, match="too few for"):
        replay_campaign(simulated.campaign, simulated.measurements, config)


def test_unknown_reference_strategy_is_rejected(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    with pytest.raises(ValueError, match="no strategy named"):
        compare_strategies(result, reference="not-a-strategy")


def test_best_selected_ignores_the_shared_initial_set(simulated, replay_config):
    """best_found can saturate on the initial reveal; best_selected cannot."""
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    frame = result.frame

    # Every strategy starts from the same revealed set, so best_found is >=
    # best_selected and can be identical across strategies even when their
    # choices differ.
    assert (frame["best_found"] >= frame["best_selected"] - 1e-9).all()

    for _, group in frame.groupby(["strategy", "seed"]):
        values = group.sort_values("round")["best_selected"].tolist()
        assert values == sorted(values)


def test_a_binding_diversity_constraint_changes_the_outcome(simulated, replay_config):
    """With a constraint that actually binds, the diversity variant must diverge."""
    from dataclasses import replace

    binding = replace(replay_config, diversity_min_distance=8)
    result = replay_campaign(simulated.campaign, simulated.measurements, binding)
    final = result.frame.sort_values("round").groupby(["strategy", "seed"]).tail(1)
    pivot = final.pivot(index="seed", columns="strategy", values="cumulative_experiments")

    plain = "assay_trained_embedding_ridge"
    diverse = f"{plain}+diversity"
    assert diverse in pivot.columns
    # The constraint cannot be relaxed to fill the batch, so fewer experiments run.
    assert (pivot[diverse] < pivot[plain]).all()


# ------------------------------------------------- the external comparison


def test_the_external_baseline_is_included_and_labelled(simulated, replay_config):
    """The benchmark has to answer 'better than an existing tool?', not only 'better
    than guessing?' — and must say the comparison is a reimplementation."""
    from assaypilot.benchmarks.baselines import EVOLVEPRO_SOURCE, EvolveProStyleMethod
    from assaypilot.features.physchem import PhysicochemicalBackend

    assert "evolvepro_style_rf" in strategy_names(replay_config, simulated.campaign)

    info = EvolveProStyleMethod(PhysicochemicalBackend(n_bins=4)).info
    assert info.kind == "external_reimplementation"
    assert any("not the authors' code" in caveat for caveat in info.caveats)
    assert any("diverse-medoid initialisation is not used" in c for c in info.caveats)
    assert EVOLVEPRO_SOURCE in info.details["source"]


def test_the_external_baseline_runs_in_a_replay(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    assert "evolvepro_style_rf" in set(result.frame["strategy"])
    rows = result.frame[result.frame["strategy"] == "evolvepro_style_rf"]
    assert rows["batch_size"].le(replay_config.batch_size).all()
    assert rows["best_selected"].notna().all()


def test_it_can_be_switched_off(simulated, replay_config):
    from dataclasses import replace

    without = replace(replay_config, include_external_baselines=False)
    assert "evolvepro_style_rf" not in strategy_names(without, simulated.campaign)


def test_the_published_defaults_are_reproduced(simulated):
    """The hyperparameters are the point of a reimplementation; pin them."""
    import numpy as np

    from assaypilot.benchmarks.baselines import EvolveProStyleMethod
    from assaypilot.features.physchem import PhysicochemicalBackend

    method = EvolveProStyleMethod(PhysicochemicalBackend(n_bins=4))
    frame = simulated.measurements.head(40)
    method.fit(frame["sequence"].tolist(), frame["value"].to_numpy(dtype=float))

    model = method._model
    assert model.n_estimators == 100
    assert model.max_depth is None
    assert model.min_samples_split == 2
    assert model.min_samples_leaf == 1
    assert model.max_features == 1.0
    assert model.bootstrap is True
    assert model.random_state == 1

    # No preprocessing, as published: predictions come straight from the embeddings.
    predictions = method.predict(frame["sequence"].tolist()[:5])
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()

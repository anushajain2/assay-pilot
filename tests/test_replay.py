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
        suite=SuiteConfig(
            representation="physchem", cache_dir=str(tmp_path / "cache"), physchem_bins=4
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
    assert {"strategy", "best_found_mean", "experiments"} <= set(summary.columns)
    assert len(summary) == len(set(result.frame["strategy"]))

    comparison = compare_strategies(result, reference="random")
    assert "random" not in set(comparison["strategy"])
    assert {"mean_difference", "n_seeds", "wins"} <= set(comparison.columns)
    assert (comparison["n_seeds"] == len(replay_config.seeds)).all()


def test_caveat_travels_with_the_result(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    assert REPLAY_CAVEAT in result.caveat
    assert "Retrospective replay" in result.to_dict()["caveat"]
    # ESM is unavailable in the test environment, which must be stated too.
    assert "not available in this environment" in result.caveat


def test_replay_refuses_an_undersized_pool(simulated):
    config = ReplayConfig(n_initial=10_000, batch_size=24, n_rounds=1, seeds=(0,))
    with pytest.raises(ValueError, match="too few for"):
        replay_campaign(simulated.campaign, simulated.measurements, config)


def test_unknown_reference_strategy_is_rejected(simulated, replay_config):
    result = replay_campaign(simulated.campaign, simulated.measurements, replay_config)
    with pytest.raises(ValueError, match="no strategy named"):
        compare_strategies(result, reference="not-a-strategy")

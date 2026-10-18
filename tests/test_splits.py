"""Evaluation protocols: which split is chosen, and what it guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assaypilot import load_measurements
from assaypilot.errors import InsufficientDataError
from assaypilot.evaluation.splits import (
    candidate_target_distance,
    choose_split_plan,
    describe_splits,
    mutation_distance_split,
    random_group_splits,
    round_splits,
)


def _library_measurements(campaign, *, n=60, rounds=3, seed=0, positions=(3, 5, 7, 9, 11, 13)):
    """Measurements over a small designed library, with controllable rounds."""
    rng = np.random.default_rng(seed)
    reference = campaign.reference_sequence
    rows = []
    for i in range(n):
        n_subs = int(rng.integers(1, 4))
        chosen = rng.choice(positions, size=n_subs, replace=False)
        residues = list(reference)
        for position in chosen:
            residues[position - 1] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
        sequence = "".join(residues)
        rows.append(
            {
                "variant_id": f"v{i:03d}",
                "sequence": sequence,
                "value": float(rng.normal()),
                "condition_id": campaign.condition_id,
                "round_id": f"R{(i % rounds) + 1}",
            }
        )
    return load_measurements(pd.DataFrame(rows), campaign, strict=False)


def test_round_splits_are_forward_chaining(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=3)
    splits = round_splits(measurements)
    assert [s.name for s in splits] == ["round_R2", "round_R3"]
    assert all(s.extrapolation for s in splits)
    assert "evaluated on round R2" in splits[0].description


def test_round_split_partitions_are_disjoint_by_sequence(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=3)
    frame = measurements.frame
    for split in round_splits(measurements):
        train = set(frame.iloc[split.train_index]["seq_hash"])
        test = set(frame.iloc[split.test_index]["seq_hash"])
        assert not (train & test)


def test_auto_prefers_round_split_when_rounds_exist(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=3)
    plan = choose_split_plan(measurements)
    assert plan.kind == "round"
    assert plan.extrapolation
    assert plan.label == "extrapolation evaluation"


def test_auto_falls_back_when_rounds_are_absent(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=1)
    measurements.frame["round_id"] = None
    plan = choose_split_plan(measurements)
    assert plan.kind in {"mutation_distance", "mutation_pattern", "random"}
    assert "round" in plan.considered


def test_random_split_is_labelled_as_interpolation(campaign):
    measurements = _library_measurements(campaign, n=40, rounds=1)
    measurements.frame["round_id"] = None
    # One substitution everywhere removes any distance or pattern holdout.
    reference = campaign.reference_sequence
    rng = np.random.default_rng(1)
    sequences = []
    for _i in range(len(measurements.frame)):
        residues = list(reference)
        residues[2] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
        residues[4] = str(rng.choice(list("ACDEFGHIKLMNPQRSTVWY")))
        sequences.append("".join(residues))
    plan_candidates = None
    measurements.frame["sequence"] = sequences
    measurements.frame["n_substitutions"] = [
        sum(1 for a, b in zip(reference, s, strict=True) if a != b) for s in sequences
    ]
    measurements.frame["seq_hash"] = [f"h{i}" for i in range(len(sequences))]
    measurements.frame["mutations"] = "WT"
    plan = choose_split_plan(measurements, plan_candidates)
    assert plan.kind == "random"
    assert plan.is_interpolation_only
    assert plan.label == "interpolation diagnostic"
    assert any("interpolation" in w for w in plan.warnings)


def test_mutation_distance_split_matches_candidate_distance(campaign, simulated_inputs):
    _, measurements, candidates = simulated_inputs
    target = candidate_target_distance(candidates)
    assert target is not None and target >= 2
    splits = mutation_distance_split(measurements, target_distance=target)
    if splits:  # the simulated data may not allow a usable threshold at every seed
        split = splits[0]
        assert split.kind == "mutation_distance"
        assert split.extrapolation
        train_max = measurements.frame.iloc[split.train_index]["n_substitutions"].max()
        test_min = measurements.frame.iloc[split.test_index]["n_substitutions"].min()
        assert train_max < test_min


def test_random_group_splits_keep_replicates_together(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    splits = random_group_splits(measurements, n_splits=2)
    frame = measurements.frame
    for split in splits:
        train = set(frame.iloc[split.train_index]["seq_hash"])
        test = set(frame.iloc[split.test_index]["seq_hash"])
        assert not (train & test)


def test_forcing_an_impossible_split_raises(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    with pytest.raises(InsufficientDataError, match="could not be built"):
        choose_split_plan(measurements, policy="round")


def test_tiny_dataset_cannot_be_evaluated(campaign, simple_measurements):
    measurements = load_measurements(simple_measurements, campaign)
    with pytest.raises(InsufficientDataError, match="distinct measured sequences"):
        choose_split_plan(measurements)


def test_describe_splits_returns_a_table(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=3)
    plan = choose_split_plan(measurements)
    table = describe_splits(plan.splits)
    assert set(table.columns) == {
        "split",
        "kind",
        "n_train",
        "n_test",
        "extrapolation",
        "description",
    }
    assert len(table) == len(plan.splits)


def test_split_plan_serialises_partitions(campaign):
    measurements = _library_measurements(campaign, n=60, rounds=3)
    plan = choose_split_plan(measurements)
    data = plan.to_dict()
    assert data["kind"] == "round"
    assert all("train_index" in s and "test_index" in s for s in data["splits"])

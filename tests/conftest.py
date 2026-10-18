"""Shared fixtures.

Every fixture is deterministic and small, and nothing here downloads model
weights: the tests use the dependency-free physicochemical representation so
the suite runs in CI without torch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from assaypilot import Campaign, SelectionSettings, load_candidates, load_measurements
from assaypilot.datasets import simulate_campaign
from assaypilot.models.registry import SuiteConfig

REFERENCE = "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNL"


@pytest.fixture
def reference() -> str:
    return REFERENCE


@pytest.fixture
def campaign() -> Campaign:
    """A minimal campaign used by the unit tests."""
    return Campaign(
        campaign_id="test-campaign",
        reference_sequence=REFERENCE,
        assay_name="initial rate",
        assay_units="s^-1",
        objective="maximize",
        condition_id="37C",
        success_threshold=1.5,
        selection=SelectionSettings(batch_size=4),
    )


def mutate(reference: str, *substitutions: tuple[int, str]) -> str:
    """Apply 1-based (position, residue) substitutions to a sequence."""
    residues = list(reference)
    for position, residue in substitutions:
        residues[position - 1] = residue
    return "".join(residues)


@pytest.fixture
def mutator():
    return mutate


@pytest.fixture
def simple_measurements(campaign: Campaign) -> pd.DataFrame:
    """A tiny hand-written measurement table with replicates and two rounds."""
    rows = [
        ("var01", mutate(REFERENCE, (3, "G")), 1.2, "R1", "rep1"),
        ("var01", mutate(REFERENCE, (3, "G")), 1.3, "R1", "rep2"),
        ("var02", mutate(REFERENCE, (5, "W")), 0.4, "R1", "rep1"),
        ("var03", mutate(REFERENCE, (7, "D")), 2.1, "R1", "rep1"),
        ("var04", mutate(REFERENCE, (3, "G"), (5, "W")), 0.8, "R2", "rep1"),
        ("var05", mutate(REFERENCE, (7, "D"), (9, "K")), 2.4, "R2", "rep1"),
    ]
    return pd.DataFrame(
        [
            {
                "variant_id": variant_id,
                "sequence": sequence,
                "value": value,
                "condition_id": "37C",
                "round_id": round_id,
                "replicate_id": replicate_id,
            }
            for variant_id, sequence, value, round_id, replicate_id in rows
        ]
    )


@pytest.fixture(scope="session")
def simulated():
    """A small simulated campaign, shared across tests that need realistic data."""
    return simulate_campaign(
        n_measured=90,
        n_candidates=120,
        n_rounds=3,
        n_variable_positions=12,
        seed=3,
    )


@pytest.fixture
def simulated_inputs(simulated):
    """``(campaign, measurements, candidates)`` loaded from the simulated campaign."""
    measurements = load_measurements(simulated.measurements, simulated.campaign, strict=True)
    candidates = load_candidates(simulated.candidates, simulated.campaign, strict=True)
    return simulated.campaign, measurements, candidates


@pytest.fixture
def physchem_suite(tmp_path) -> SuiteConfig:
    """A suite that needs no model weights, with a per-test embedding cache."""
    return SuiteConfig(
        representation="physchem",
        cache_dir=str(tmp_path / "cache"),
        physchem_bins=4,
    )

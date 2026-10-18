"""AssayPilot — plan the next round of protein variant experiments.

AssayPilot takes the measurements a campaign has already collected and a pool
of untested candidates, works out which prediction method the available
evidence actually supports, and selects a fixed-size batch to measure next.

A minimal run::

    from assaypilot import Campaign, load_candidates, load_measurements, plan_batch
    from assaypilot.reporting import export_plan

    campaign = Campaign(
        campaign_id="demo",
        reference_sequence="MKT...",
        assay_name="initial rate",
        assay_units="s^-1",
        objective="maximize",
        condition_id="37C_pH7.4",
    )
    measurements = load_measurements("measurements.csv", campaign)
    candidates = load_candidates("candidates.csv", campaign)
    plan = plan_batch(campaign, candidates, measurements)
    export_plan(plan, "out/")

What the tool claims, and what it does not, is set out in ``docs/claims.md``.
In short: every predictor is tied to one named assay under one condition;
evidence comes from held-out measurements of that campaign; and when the
evidence cannot separate the methods, the plan says so instead of naming a
winner.
"""

from __future__ import annotations

from .campaign import Campaign, SelectionSettings
from .data.loaders import (
    CandidateSet,
    MeasurementSet,
    apply_transform,
    load_candidates,
    load_measurements,
)
from .errors import (
    AssayPilotError,
    BackendUnavailableError,
    CampaignError,
    InputError,
    InsufficientDataError,
    SequenceError,
)
from .evaluation.compare import EvaluationResult, evaluate_methods
from .evaluation.splits import choose_split_plan
from .models.registry import SuiteConfig, build_methods
from .pipeline import ExperimentPlan, plan_batch
from .scoring import RoundScorecard, score_round
from .store import CampaignStore

__all__ = [
    "AssayPilotError",
    "BackendUnavailableError",
    "Campaign",
    "CampaignError",
    "CampaignStore",
    "CandidateSet",
    "EvaluationResult",
    "ExperimentPlan",
    "InputError",
    "InsufficientDataError",
    "MeasurementSet",
    "RoundScorecard",
    "SelectionSettings",
    "SequenceError",
    "SuiteConfig",
    "apply_transform",
    "build_methods",
    "choose_split_plan",
    "evaluate_methods",
    "load_candidates",
    "load_measurements",
    "plan_batch",
    "score_round",
]

try:  # pragma: no cover - depends on installation method
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("assaypilot")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0+source"

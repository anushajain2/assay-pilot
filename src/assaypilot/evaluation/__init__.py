"""Splits, metrics, method comparison, candidate coverage, and rank variability."""

from .compare import (
    EvaluationResult,
    MethodSummary,
    PairedComparison,
    Recommendation,
    evaluate_methods,
)
from .coverage import CoverageReport, assess_coverage
from .metrics import best_at_k, compute_metrics, hit_rate_at_k, random_reference, spearman
from .splits import Split, SplitPlan, choose_split_plan, describe_splits
from .uncertainty import RankVariability, bootstrap_rank_variability

__all__ = [
    "CoverageReport",
    "EvaluationResult",
    "MethodSummary",
    "PairedComparison",
    "RankVariability",
    "Recommendation",
    "Split",
    "SplitPlan",
    "assess_coverage",
    "best_at_k",
    "bootstrap_rank_variability",
    "choose_split_plan",
    "compute_metrics",
    "describe_splits",
    "evaluate_methods",
    "hit_rate_at_k",
    "random_reference",
    "spearman",
]

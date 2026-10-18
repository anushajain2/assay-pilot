"""The planning pipeline: measurements and candidates in, an experiment plan out.

This module is the one place where the whole chain is assembled, in a fixed
order:

1. validate the inputs against the campaign,
2. choose an evaluation protocol that matches the next experimental step,
3. compare the fixed method suite on exactly the same data,
4. apply the stated selection rule to decide which method the evidence
   supports,
5. refit that method on everything measured and score the candidates,
6. check how well the evidence covers those candidates,
7. filter for eligibility, rank, and apply a selection policy,
8. record all of it in a run manifest.

The same function serves the command line, the Streamlit interface, and the
benchmark replay harness, so there is only one implementation of the decision
to audit.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .campaign import Campaign, SelectionSettings
from .data.loaders import CandidateSet, MeasurementSet
from .data.validation import ValidationReport
from .errors import InsufficientDataError
from .evaluation.compare import EvaluationResult, evaluate_methods
from .evaluation.coverage import CoverageReport, assess_coverage
from .evaluation.splits import SplitPlan, choose_split_plan
from .evaluation.uncertainty import RankVariability, bootstrap_rank_variability
from .models.base import Method
from .models.registry import (
    SuiteConfig,
    build_cache,
    build_methods,
    excluded_by_config,
    split_by_availability,
)
from .reporting.manifest import RunManifest, environment_fingerprint, new_run_id
from .selection.filters import FilterResult, filter_candidates
from .selection.policies import SelectionOutcome, build_policy
from .sequences import min_distance_to_set

logger = logging.getLogger(__name__)

#: Evidence levels, in decreasing strength.
EVIDENCE_SUPPORTED = "supported"
EVIDENCE_EXPLORATORY = "exploratory"
EVIDENCE_NONE = "none"


def design_id(campaign: Campaign, sequence: str) -> str:
    """A stable, immutable identifier for a designed construct.

    Derived from the campaign id and the sequence, so the same sequence always
    receives the same identifier — including in a later round, which is what
    makes measured results matchable back to the plan that proposed them.
    """
    blob = f"{campaign.campaign_id}|{sequence}".encode()
    return "d" + hashlib.sha256(blob).hexdigest()[:11]


@dataclass
class ExperimentPlan:
    """A batch to measure next, with the evidence that supports it."""

    campaign: Campaign
    run_id: str
    candidates: CandidateSet
    measurements: MeasurementSet | None
    ranking: pd.DataFrame
    selected: pd.DataFrame
    outcome: SelectionOutcome
    filter_result: FilterResult
    coverage: CoverageReport
    evaluation: EvaluationResult | None
    method_key: str
    method_name: str
    evidence_level: str
    manifest: RunManifest
    rank_variability: RankVariability | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    settings: SelectionSettings = field(default_factory=SelectionSettings)

    @property
    def is_exploratory(self) -> bool:
        return self.evidence_level != EVIDENCE_SUPPORTED

    @property
    def budget(self) -> dict[str, int]:
        """Candidate wells, control wells, and the total, stated separately."""
        settings = self.settings
        return {
            "candidates_selected": int(len(self.selected)),
            "candidates_requested": int(settings.batch_size),
            "replicates_per_candidate": int(settings.replicates_per_candidate),
            "candidate_wells": int(len(self.selected) * settings.replicates_per_candidate),
            "control_wells": int(len(settings.control_sequences)),
            "total_wells": int(
                len(self.selected) * settings.replicates_per_candidate
                + len(settings.control_sequences)
            ),
        }

    def headline(self) -> str:
        """One sentence stating what was chosen and on what evidence."""
        qualifier = {
            EVIDENCE_SUPPORTED: "supported by held-out evaluation",
            EVIDENCE_EXPLORATORY: "exploratory: the evaluation does not support a reliable"
            " ordering of methods",
            EVIDENCE_NONE: "exploratory: no measured data was available to validate any"
            " method on this assay",
        }[self.evidence_level]
        return (
            f"{len(self.selected)} of {self.outcome.requested} requested candidates selected"
            f" using {self.method_name} with the {self.outcome.policy} policy — {qualifier}."
        )


def _refit_on_all(
    method: Method, measurements: MeasurementSet, *, sign: int
) -> tuple[Method, np.ndarray, np.ndarray, list[str]]:
    """Refit the chosen method on every usable measurement."""
    frame = measurements.frame
    sequences = frame["sequence"].tolist()
    y = sign * frame["value"].to_numpy(dtype=float)
    groups = frame["seq_hash"].to_numpy()
    fitted = method.clone()
    fitted.fit(sequences, y, groups=groups)
    return fitted, y, groups, sequences


def _evaluated_distances(measurements: MeasurementSet, split_plan: SplitPlan) -> list[int]:
    """How far each held-out sequence was from *its own split's* training set.

    Taking the union of training rows across splits would put a sequence in
    both partitions — a variant held out in round 2 is training data for the
    round 3 split — and its distance to "training" would come out as zero. The
    region the evidence covers would then look far narrower than it is, and
    candidates that were in fact covered would be flagged as outside it.
    """
    frame = measurements.frame
    distances: list[int] = []
    for split in split_plan.splits:
        training = frame.iloc[split.train_index]["sequence"].tolist()
        if not training:
            continue
        for sequence in frame.iloc[split.test_index]["sequence"]:
            distances.append(min_distance_to_set(sequence, training))
    return distances


def plan_batch(
    campaign: Campaign,
    candidates: CandidateSet,
    measurements: MeasurementSet | None = None,
    *,
    settings: SelectionSettings | None = None,
    suite: SuiteConfig | None = None,
    policy: str = "top_ranked",
    split_policy: str = "auto",
    restrict_to_evaluated_range: bool = False,
    n_bootstrap: int = 500,
    rank_variability_refits: int = 0,
    seed: int = 0,
    run_id: str | None = None,
    input_checksums: dict[str, Any] | None = None,
) -> ExperimentPlan:
    """Produce the next batch to measure, with the evidence behind it.

    Parameters
    ----------
    measurements:
        ``None`` or an empty set puts the run in exploratory mode: no method
        can be validated on this assay, so the label-free pretrained score is
        used for ranking and the plan is labelled accordingly.
    policy:
        ``"top_ranked"`` (default) or ``"diversity"``. The diversity policy
        needs ``settings.diversity_min_distance``.
    restrict_to_evaluated_range:
        Exclude candidates further from the measured variants than anything the
        evaluation covered, instead of only flagging them.
    rank_variability_refits:
        Number of bootstrap refits used to report ranking stability. ``0``
        skips it, which keeps a plan fast on large pools.
    """
    settings = settings or campaign.selection
    suite = suite or SuiteConfig()
    run_id = run_id or new_run_id()
    warnings: list[str] = []
    notes: list[str] = []

    if candidates.frame.empty:
        raise InsufficientDataError(
            "the candidate pool is empty after validation; nothing can be selected"
        )

    cache = build_cache(suite)
    all_methods = build_methods(campaign, suite, cache=cache)
    usable, unavailable = split_by_availability(all_methods)
    # A method left out by the configuration is reported in the same place as
    # one whose backend is missing, with its own wording: the reader needs to
    # know the comparison did not include it either way.
    unavailable = {**excluded_by_config(suite), **unavailable}
    for key, reason in unavailable.items():
        warnings.append(f"{key} was not included: {reason}")

    evaluation: EvaluationResult | None = None
    split_plan: SplitPlan | None = None
    evidence_level = EVIDENCE_NONE
    chosen_method: Method | None = None

    # Censored and non-finite measurements are removed here, not by the caller.
    # Leaving it to the caller meant the CLI was safe while the documented
    # Python API and the interface quietly fitted on detection limits as though
    # they were exact measurements. The removed rows are recorded as
    # exclusions, so they still appear in the report and in exclusions.csv.
    if measurements is not None:
        measurements = measurements.for_regression()

    has_labels = measurements is not None and not measurements.frame.empty

    if has_labels:
        assert measurements is not None
        try:
            split_plan = choose_split_plan(measurements, candidates, policy=split_policy, seed=seed)
        except InsufficientDataError as exc:
            warnings.append(
                f"No held-out evaluation could be built: {exc} The plan is exploratory."
            )
            split_plan = None

    if has_labels and split_plan is not None:
        assert measurements is not None
        evaluation = evaluate_methods(
            usable,
            measurements,
            split_plan,
            campaign=campaign,
            k=settings.batch_size,
            n_bootstrap=n_bootstrap,
            seed=seed,
            unavailable=unavailable,
        )
        evidence_level = (
            EVIDENCE_SUPPORTED
            if evaluation.recommendation.evidence_level == "supported"
            else EVIDENCE_EXPLORATORY
        )
        chosen_key = evaluation.recommendation.method_key
        chosen_method = next(m for m in usable if m.info.key == chosen_key)
        warnings.extend(split_plan.warnings)
    elif has_labels:
        # Labels exist but no split could be built: fall back to the simplest
        # label-using method and say the ranking is unvalidated.
        evidence_level = EVIDENCE_EXPLORATORY
        label_using = [m for m in usable if m.info.uses_labels]
        chosen_method = min(label_using, key=lambda m: m.info.complexity) if label_using else None

    if chosen_method is None:
        label_free = [m for m in usable if not m.info.uses_labels]
        if not label_free:
            raise InsufficientDataError(
                "No measured data is available and the label-free pretrained score could not"
                " be used"
                + (f" ({'; '.join(unavailable.values())})" if unavailable else "")
                + ". Add measurements, or install the ESM extra to rank candidates with the"
                " pretrained model."
            )
        chosen_method = label_free[0]
        evidence_level = EVIDENCE_NONE if not has_labels else evidence_level
        notes.append(
            "Ranking uses the label-free pretrained score. No claim of assay-specific"
            " validation is made for this plan."
        )

    # Refit the chosen method on every usable measurement, then score candidates.
    sign = campaign.sign
    fit_details: dict[str, Any] = {}
    if has_labels:
        assert measurements is not None
        fitted, y_all, groups_all, train_sequences = _refit_on_all(
            chosen_method, measurements, sign=sign
        )
        fit_details = fitted.fit_details()
    else:
        # Label-free method: fit is a no-op, present only for interface symmetry.
        fitted = chosen_method.clone()
        fitted.fit([], np.array([]))
        y_all = np.array([])
        groups_all = np.array([])
        train_sequences = []

    candidate_sequences = candidates.frame["sequence"].tolist()
    scores = np.asarray(fitted.predict(candidate_sequences), dtype=float)

    ranking = candidates.frame.copy()
    ranking["score"] = scores
    # Predicted assay values only exist for methods that regress on the measured
    # outcome. A pretrained score is a score, and the column stays empty.
    if chosen_method.info.uses_labels and has_labels:
        ranking["predicted_value"] = sign * scores
    else:
        ranking["predicted_value"] = np.nan
    # The units follow the value space the model was fitted in: a prediction
    # made on log10 values is not a number of RFU.
    ranking["predicted_value_units"] = (
        measurements.value_units()
        if chosen_method.info.uses_labels and has_labels and measurements is not None
        else ""
    )

    # Observed values, where a candidate has already been measured.
    if has_labels:
        assert measurements is not None
        observed = measurements.frame.groupby("seq_hash")["value"].mean().rename("observed_value")
        ranking = ranking.merge(observed, on="seq_hash", how="left")
    else:
        ranking["observed_value"] = np.nan

    # Coverage of the candidates by the evidence.
    if has_labels and split_plan is not None:
        assert measurements is not None
        coverage = assess_coverage(
            candidates,
            measurements,
            evaluated_distances=_evaluated_distances(measurements, split_plan),
        )
    elif has_labels:
        assert measurements is not None
        coverage = assess_coverage(candidates, measurements)
    else:
        empty = MeasurementSet(
            campaign=campaign,
            frame=pd.DataFrame(columns=["sequence", "seq_hash", "value", "n_substitutions"]),
            report=ValidationReport(source="<no measurements>"),
        )
        coverage = assess_coverage(candidates, empty)
    warnings.extend(coverage.warnings)

    ranking = ranking.merge(
        coverage.frame[
            [
                "candidate_id",
                "min_distance_to_measured",
                "n_unseen_substitutions",
                "within_evaluated_range",
            ]
        ],
        on="candidate_id",
        how="left",
    )

    # Eligibility filtering, then ranking, then the policy.
    allowed_ids = None
    if restrict_to_evaluated_range:
        allowed_ids = coverage.frame.loc[
            coverage.frame["within_evaluated_range"], "candidate_id"
        ].tolist()
        notes.append(
            "Selection was restricted to candidates inside the evaluated region at the"
            " researcher's request."
        )

    filter_result = filter_candidates(
        candidates,
        campaign,
        measurements=measurements,
        settings=settings,
        restrict_to_candidate_ids=allowed_ids,
    )

    eligible_ids = set(filter_result.eligible["candidate_id"])
    ranking["eligible"] = ranking["candidate_id"].isin(eligible_ids)
    ranked_eligible = (
        ranking[ranking["eligible"]]
        .sort_values(["score", "candidate_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking = ranking.sort_values(
        ["eligible", "score", "candidate_id"], ascending=[False, False, True]
    ).reset_index(drop=True)
    rank_by_id = {
        candidate_id: position
        for position, candidate_id in enumerate(ranked_eligible["candidate_id"], start=1)
    }
    ranking["rank_among_eligible"] = ranking["candidate_id"].map(rank_by_id)

    if ranked_eligible.empty:
        raise InsufficientDataError(
            "no candidate passed the eligibility filters; relax the constraints or supply a"
            " different pool. The exclusion reasons are in the filter report."
        )

    selection_policy = build_policy(policy, min_distance=settings.diversity_min_distance)
    outcome = selection_policy.select(ranked_eligible, settings.batch_size)
    if outcome.shortfall:
        warnings.append(
            f"The batch is short by {outcome.shortfall} candidate(s): {outcome.shortfall_reason}"
        )

    selected = outcome.selected.copy()
    selected.insert(0, "design_id", [design_id(campaign, s) for s in selected["sequence"]])
    selected["run_id"] = run_id
    selected["campaign_id"] = campaign.campaign_id
    selected["method"] = chosen_method.info.key
    selected["evidence_level"] = evidence_level

    # Ranking stability, when asked for and meaningful.
    rank_variability: RankVariability | None = None
    if rank_variability_refits > 0 and has_labels and chosen_method.info.uses_labels:
        rank_variability = bootstrap_rank_variability(
            chosen_method,
            train_sequences=train_sequences,
            y=y_all,
            groups=groups_all,
            candidate_ids=ranked_eligible["candidate_id"].tolist(),
            candidate_sequences=ranked_eligible["sequence"].tolist(),
            top_k=settings.batch_size,
            n_bootstrap=rank_variability_refits,
            seed=seed,
        )
        selected = selected.merge(
            rank_variability.frame[["candidate_id", "rank_std", "top_k_frequency"]],
            on="candidate_id",
            how="left",
        )

    # The representation is recorded at the top level even when the chosen
    # method does not use it: the comparison that picked the simpler method
    # still depended on the ESM results, so reproducing the decision needs the
    # checkpoint and revision those results came from.
    representation: dict[str, Any] | None = None
    for candidate_method in all_methods:
        details = candidate_method.info.details.get("representation")
        if details:
            representation = details

    manifest = _build_manifest(
        run_id=run_id,
        campaign=campaign,
        representation=representation,
        settings=settings,
        suite=suite,
        policy=policy,
        split_policy=split_policy,
        seed=seed,
        n_bootstrap=n_bootstrap,
        measurements=measurements,
        candidates=candidates,
        evaluation=evaluation,
        coverage=coverage,
        filter_result=filter_result,
        outcome=outcome,
        ranking=ranking,
        selected=selected,
        method=chosen_method,
        fit_details=fit_details,
        evidence_level=evidence_level,
        warnings=warnings,
        notes=notes,
        rank_variability=rank_variability,
        input_checksums=input_checksums,
        cache=cache,
    )

    return ExperimentPlan(
        campaign=campaign,
        run_id=run_id,
        candidates=candidates,
        measurements=measurements,
        ranking=ranking,
        selected=selected,
        outcome=outcome,
        filter_result=filter_result,
        coverage=coverage,
        evaluation=evaluation,
        method_key=chosen_method.info.key,
        method_name=chosen_method.info.name,
        evidence_level=evidence_level,
        manifest=manifest,
        rank_variability=rank_variability,
        warnings=warnings,
        notes=notes,
        settings=settings,
    )


def _build_manifest(
    *,
    run_id: str,
    campaign: Campaign,
    settings: SelectionSettings,
    suite: SuiteConfig,
    representation: dict[str, Any] | None,
    policy: str,
    split_policy: str,
    seed: int,
    n_bootstrap: int,
    measurements: MeasurementSet | None,
    candidates: CandidateSet,
    evaluation: EvaluationResult | None,
    coverage: CoverageReport,
    filter_result: FilterResult,
    outcome: SelectionOutcome,
    ranking: pd.DataFrame,
    selected: pd.DataFrame,
    method: Method,
    fit_details: dict[str, Any],
    evidence_level: str,
    warnings: Sequence[str],
    notes: Sequence[str],
    rank_variability: RankVariability | None,
    input_checksums: dict[str, Any] | None,
    cache: Any,
) -> RunManifest:
    """Assemble the manifest for a completed run."""
    prediction_columns = [
        "candidate_id",
        "mutations",
        "score",
        "predicted_value",
        "observed_value",
        "eligible",
        "rank_among_eligible",
        "min_distance_to_measured",
        "n_unseen_substitutions",
        "within_evaluated_range",
    ]
    predictions = ranking[prediction_columns].to_dict(orient="records")

    exclusions = [e.to_dict() for e in filter_result.exclusions]
    if measurements is not None:
        exclusions = [e.to_dict() for e in measurements.report.exclusions] + exclusions
    exclusions = [e.to_dict() for e in candidates.report.exclusions] + exclusions

    data_section: dict[str, Any] = {
        "candidates": {**candidates.to_dict(), "report": candidates.report.to_dict()},
    }
    if measurements is not None:
        data_section["measurements"] = {
            **measurements.to_dict(),
            "report": measurements.report.to_dict(),
        }
    else:
        data_section["measurements"] = None

    partitions = (
        {
            split.name: {
                "train_index": [int(i) for i in split.train_index],
                "test_index": [int(i) for i in split.test_index],
            }
            for split in evaluation.split_plan.splits
        }
        if evaluation is not None
        else {}
    )

    return RunManifest(
        run_id=run_id,
        campaign=campaign.to_dict(),
        environment=environment_fingerprint(),
        inputs=dict(input_checksums or {}),
        config={
            "selection_settings": settings.to_dict(),
            "suite": suite.to_dict(),
            "representation": representation,
            "selection_policy": policy,
            "split_policy": split_policy,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "embedding_cache_enabled": bool(getattr(cache, "enabled", False)),
            "method_used": method.info.to_dict(),
            "fit_details": fit_details,
        },
        runtime={"embedding_cache": cache.stats() if hasattr(cache, "stats") else {}},
        data=data_section,
        evaluation={
            **(evaluation.to_dict() if evaluation is not None else {}),
            "evidence_level": evidence_level,
            "partitions": partitions,
            "rank_variability": rank_variability.to_dict() if rank_variability else None,
        },
        coverage=coverage.to_dict(),
        selection={
            **outcome.to_dict(),
            "filters": filter_result.to_dict(),
            "budget": {
                "candidates_selected": int(len(selected)),
                "candidates_requested": int(settings.batch_size),
                "replicates_per_candidate": int(settings.replicates_per_candidate),
                "control_wells": int(len(settings.control_sequences)),
                "total_wells": int(
                    len(selected) * settings.replicates_per_candidate
                    + len(settings.control_sequences)
                ),
            },
        },
        predictions=predictions,
        exclusions=exclusions,
        selected=selected.to_dict(orient="records"),
        warnings=list(warnings),
        notes=list(notes),
    )

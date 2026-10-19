"""Scoring a previous plan against the results it produced.

When a new round of measurements arrives, AssayPilot scores the predictions it
made **before** retraining anything. This is the only prospective evidence the
tool ever has: everything else is retrospective.

Matching is by ``design_id``, the immutable identifier written into the plan.
It is recomputed from each measured sequence, so a lab that renames its wells
still matches: identity travels with the construct, not with the label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .campaign import Campaign
from .data.loaders import MeasurementSet
from .evaluation.metrics import spearman
from .pipeline import design_id
from .reporting.manifest import RunManifest


@dataclass
class RoundScorecard:
    """How a plan's predictions held up against the measurements that followed."""

    run_id: str
    campaign_id: str
    matched: pd.DataFrame
    n_planned: int
    n_matched: int
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the planned batch that has a measured result."""
        return self.n_matched / self.n_planned if self.n_planned else float("nan")

    def summary_lines(self) -> list[str]:
        lines = [
            f"Run {self.run_id}: {self.n_matched} of {self.n_planned} planned candidates have"
            " measured results."
        ]
        if self.n_matched == 0:
            lines.append("No predictions could be scored yet.")
            return lines
        rank = self.metrics.get("spearman")
        if rank is not None and np.isfinite(rank):
            lines.append(
                f"Rank correlation between predicted and measured outcome in this batch:"
                f" {rank:.3f}."
            )
        else:
            lines.append(
                "Rank correlation could not be computed for this batch (too few matched"
                " results, or the method produced no ordering)."
            )
        if self.metrics.get("hit_rate") is not None:
            lines.append(
                f"Hit rate in the delivered batch: {self.metrics['hit_rate']:.0%}"
                f" ({self.metrics['n_hits']} of {self.n_matched})."
            )
        if self.metrics.get("best_observed") is not None:
            lines.append(
                f"Best measured value in the batch: {self.metrics['best_observed']:.4g}"
                + (
                    f", previous campaign best {self.metrics['previous_best']:.4g}."
                    if self.metrics.get("previous_best") is not None
                    else "."
                )
            )
        lines.extend(self.notes)
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "campaign_id": self.campaign_id,
            "n_planned": self.n_planned,
            "n_matched": self.n_matched,
            "coverage": self.coverage,
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


def _planned_frame(manifest: RunManifest) -> pd.DataFrame:
    planned = pd.DataFrame(manifest.selected)
    if planned.empty:
        raise ValueError(f"run {manifest.run_id} selected no variants, so nothing can be scored")
    return planned


def score_round(
    manifest: RunManifest | str | Path,
    new_measurements: MeasurementSet,
    *,
    campaign: Campaign | None = None,
    previous_measurements: MeasurementSet | None = None,
) -> RoundScorecard:
    """Compare a plan's predictions with the measurements that came back.

    Parameters
    ----------
    manifest:
        The run manifest produced by :func:`~assaypilot.pipeline.plan_batch`,
        or a path to it.
    new_measurements:
        The new round. Only rows matching the planned batch are scored.
    previous_measurements:
        Optional earlier rounds, used to report whether the batch improved on
        the campaign's previous best.
    """
    if not isinstance(manifest, RunManifest):
        manifest = RunManifest.read(manifest)
    campaign = campaign or Campaign.from_dict(manifest.campaign)

    planned = _planned_frame(manifest)
    n_planned = len(planned)

    observed = new_measurements.frame.copy()
    observed["design_id"] = [design_id(campaign, s) for s in observed["sequence"]]
    observed_mean = (
        observed.groupby("design_id")
        .agg(
            # Deliberately not "observed_value": the planned frame already has
            # that column (empty for unmeasured candidates), and a merge would
            # silently keep the empty one.
            measured_value=("value", "mean"),
            n_measurements=("value", "size"),
            variant_id=("variant_id", "first"),
            round_id=("round_id", "first"),
        )
        .reset_index()
    )

    matched = planned.drop(columns=["observed_value"], errors="ignore").merge(
        observed_mean, on="design_id", how="inner"
    )

    notes: list[str] = []
    unmatched = n_planned - len(matched)
    if unmatched:
        notes.append(
            f"{unmatched} planned candidate(s) have no measured result in this round. They are"
            " not counted in the scores."
        )

    metrics: dict[str, Any] = {}
    if not matched.empty:
        sign = campaign.sign
        predicted = matched["score"].to_numpy(dtype=float)
        actual = sign * matched["measured_value"].to_numpy(dtype=float)
        metrics["spearman"] = spearman(actual, predicted)
        metrics["n_matched"] = int(len(matched))
        metrics["best_observed"] = float(
            matched["measured_value"].max()
            if campaign.objective == "maximize"
            else matched["measured_value"].min()
        )
        metrics["mean_observed"] = float(matched["measured_value"].mean())

        # The threshold is mapped into whatever value space the new round was
        # loaded in, so a transformed round is not compared against a raw
        # threshold.
        hit_flags = new_measurements.is_hit_array(
            matched["measured_value"].to_numpy(dtype=float), campaign=campaign
        )
        if hit_flags is not None:
            metrics["n_hits"] = int(hit_flags.sum())
            metrics["hit_rate"] = float(hit_flags.mean())
            metrics["threshold_used"] = new_measurements.threshold_in_value_space(campaign)
            metrics["threshold_space"] = new_measurements.value_units()
        elif campaign.success_threshold is None:
            notes.append(
                "No success threshold is defined for this campaign, so no hit rate is"
                " reported. Define one before the next round if you want that number."
            )
        else:
            notes.append(
                f"The success threshold ({campaign.success_threshold:g}"
                f" {campaign.assay_units}) cannot be expressed on the"
                f" {new_measurements.transform}-transformed scale these measurements use,"
                " so no hit rate is reported."
            )

        if previous_measurements is not None and not previous_measurements.frame.empty:
            previous_values = previous_measurements.frame["value"]
            metrics["previous_best"] = float(
                previous_values.max() if campaign.objective == "maximize" else previous_values.min()
            )
            improved = (
                metrics["best_observed"] > metrics["previous_best"]
                if campaign.objective == "maximize"
                else metrics["best_observed"] < metrics["previous_best"]
            )
            metrics["improved_on_previous_best"] = bool(improved)

        if len(matched) < 5:
            notes.append(
                f"Only {len(matched)} matched result(s): these numbers describe this batch and"
                " should not be read as a performance estimate for the method."
            )

    notes.append(
        "This scorecard compares predictions with the measurements that followed them. It does"
        " not say what a different selection would have produced, because those variants were"
        " not measured."
    )

    columns = [
        c
        for c in (
            "design_id",
            "candidate_id",
            "selection_rank",
            "mutations",
            "score",
            "predicted_value",
            "measured_value",
            "n_measurements",
            "round_id",
        )
        if c in matched.columns
    ]

    return RoundScorecard(
        run_id=manifest.run_id,
        campaign_id=campaign.campaign_id,
        matched=matched[columns] if columns else matched,
        n_planned=n_planned,
        n_matched=int(len(matched)),
        metrics=metrics,
        notes=notes,
    )

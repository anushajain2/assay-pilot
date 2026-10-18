"""Exports that work without AssayPilot.

Everything a plan produces is written as plain CSV, FASTA, JSON, and HTML, so
a collaborator can order the batch, load it into a LIMS, or audit the decision
without installing anything.

The columns keep three categories strictly apart:

``observed_value``
    measured in the lab,
``predicted_value`` / ``score``
    produced by a model — ``predicted_value`` only exists when the method
    regresses on the measured outcome, while ``score`` may be an arbitrary
    scale,
``selection_reason``
    why this candidate is in the batch.

A confident-looking number never changes category on its way into a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..data.fasta import FastaRecord, write_fasta
from ..data.validation import exclusions_frame

if TYPE_CHECKING:  # avoids a circular import: pipeline builds the manifest this module writes
    from ..pipeline import ExperimentPlan

#: Columns written to ``selected_variants.csv``, in order.
SELECTED_COLUMNS: tuple[str, ...] = (
    "design_id",
    "candidate_id",
    "selection_rank",
    "sequence",
    "mutations",
    "n_substitutions",
    "score",
    "predicted_value",
    "predicted_value_units",
    "observed_value",
    "min_distance_to_measured",
    "n_unseen_substitutions",
    "within_evaluated_range",
    "rank_std",
    "top_k_frequency",
    "selection_reason",
    "method",
    "evidence_level",
    "campaign_id",
    "run_id",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "sequence",
    "mutations",
    "n_substitutions",
    "eligible",
    "rank_among_eligible",
    "score",
    "predicted_value",
    "predicted_value_units",
    "observed_value",
    "min_distance_to_measured",
    "n_unseen_substitutions",
    "within_evaluated_range",
)


def _ordered(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    present = [c for c in columns if c in frame.columns]
    extra = [c for c in frame.columns if c not in columns and c not in {"seq_hash", "source_row"}]
    return frame[present + extra]


def export_plan(
    plan: ExperimentPlan,
    output_dir: str | Path,
    *,
    write_report: bool = True,
    prefix: str = "",
) -> dict[str, Path]:
    """Write every artifact for a plan into ``output_dir``.

    Returns a mapping from artifact name to the path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def path_for(name: str) -> Path:
        return output_dir / f"{prefix}{name}"

    selected = _ordered(plan.selected, SELECTED_COLUMNS)
    selected_path = path_for("selected_variants.csv")
    selected.to_csv(selected_path, index=False)
    written["selected_variants"] = selected_path

    fasta_path = path_for("selected_variants.fasta")
    records = [
        FastaRecord(
            identifier=str(row.design_id),
            sequence=str(row.sequence),
            description=(
                f"candidate_id={row.candidate_id} rank={row.selection_rank}"
                f" mutations={row.mutations} campaign={plan.campaign.campaign_id}"
                f" run={plan.run_id}"
            ),
        )
        for row in selected.itertuples(index=False)
    ]
    write_fasta(fasta_path, records)
    written["selected_fasta"] = fasta_path

    predictions_path = path_for("all_predictions.csv")
    _ordered(plan.ranking, PREDICTION_COLUMNS).to_csv(predictions_path, index=False)
    written["all_predictions"] = predictions_path

    exclusion_rows = list(plan.candidates.report.exclusions) + list(plan.filter_result.exclusions)
    if plan.measurements is not None:
        exclusion_rows = list(plan.measurements.report.exclusions) + exclusion_rows
    exclusions_path = path_for("exclusions.csv")
    exclusions_frame(exclusion_rows).to_csv(exclusions_path, index=False)
    written["exclusions"] = exclusions_path

    manifest_path = path_for("run_manifest.json")
    plan.manifest.write(manifest_path)
    written["manifest"] = manifest_path

    campaign_path = path_for("campaign.json")
    plan.campaign.to_json(campaign_path)
    written["campaign"] = campaign_path

    if plan.coverage.frame is not None and not plan.coverage.frame.empty:
        coverage_path = path_for("candidate_coverage.csv")
        plan.coverage.frame.to_csv(coverage_path, index=False)
        written["coverage"] = coverage_path

    if plan.evaluation is not None:
        metrics_path = path_for("method_comparison.csv")
        method_comparison_frame(plan).to_csv(metrics_path, index=False)
        written["method_comparison"] = metrics_path

    if write_report:
        from .html import render_report

        report_path = path_for("report.html")
        report_path.write_text(render_report(plan), encoding="utf-8")
        written["report"] = report_path

    return written


def method_comparison_frame(plan: ExperimentPlan) -> pd.DataFrame:
    """Tidy table of per-method, per-split metrics plus the aggregate row."""
    if plan.evaluation is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for key, summary in plan.evaluation.summaries.items():
        for split_name, metrics in summary.per_split.items():
            rows.append(
                {
                    "method": key,
                    "method_name": summary.info.name,
                    "split": split_name,
                    "scope": "split",
                    "n_test": metrics.n_test,
                    "k": metrics.k,
                    **metrics.values,
                }
            )
        rows.append(
            {
                "method": key,
                "method_name": summary.info.name,
                "split": "mean across splits",
                "scope": "aggregate",
                "n_test": sum(m.n_test for m in summary.per_split.values()),
                "k": plan.evaluation.k,
                **summary.mean_metrics,
            }
        )
    return pd.DataFrame(rows)


def budget_frame(plan: ExperimentPlan) -> pd.DataFrame:
    """Candidate wells, control wells, and the total, stated separately."""
    budget = plan.budget
    return pd.DataFrame(
        [
            {"item": "candidates requested", "count": budget["candidates_requested"]},
            {"item": "candidates selected", "count": budget["candidates_selected"]},
            {"item": "replicates per candidate", "count": budget["replicates_per_candidate"]},
            {"item": "candidate wells", "count": budget["candidate_wells"]},
            {"item": "control wells", "count": budget["control_wells"]},
            {"item": "total wells", "count": budget["total_wells"]},
        ]
    )

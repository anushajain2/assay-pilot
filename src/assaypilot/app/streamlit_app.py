"""The AssayPilot interface.

Launch it with ``assaypilot app`` (needs ``pip install "assaypilot[app]"``).

Four screens, in the order the work happens:

1. **Campaign** — declare what is being measured, under which condition, and
   with what budget.
2. **Data review** — what the files contained, what was excluded and why,
   which replicates disagree, and which evaluation protocol the data supports.
3. **Model evidence** — the comparison, in plain language, with the rule that
   chose a method and the limits of what it supports.
4. **Batch and export** — the selected candidates, the reasons, and the files.

The screens call exactly the same functions as the command line, so a plan
made here and a plan made there are the same plan with the same manifest.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from assaypilot import (
    Campaign,
    SelectionSettings,
    load_candidates,
    load_measurements,
    plan_batch,
    score_round,
)
from assaypilot.data.loaders import apply_transform
from assaypilot.data.validation import exclusions_frame
from assaypilot.errors import AssayPilotError
from assaypilot.evaluation.metrics import HUMAN_NAMES
from assaypilot.evaluation.splits import choose_split_plan, describe_splits
from assaypilot.evaluation.uncertainty import RANK_VARIABILITY_CAVEAT
from assaypilot.features.esm import DEFAULT_CHECKPOINT, esm_available
from assaypilot.models.registry import SuiteConfig
from assaypilot.reporting.exports import export_plan, method_comparison_frame
from assaypilot.reporting.html import render_report

st.set_page_config(page_title="AssayPilot", page_icon="🧪", layout="wide")

EVIDENCE_STYLE = {
    "supported": ("✅", "Supported by held-out evaluation"),
    "exploratory": ("⚠️", "Exploratory — the evidence does not support a reliable ordering"),
    "none": ("⚠️", "Exploratory — no assay-specific validation"),
}


# ----------------------------------------------------------------- utilities


def _state(key: str, default: Any = None) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _read_upload(upload) -> pd.DataFrame | Path | None:
    """Turn an uploaded file into something the loaders accept."""
    if upload is None:
        return None
    name = upload.name.lower()
    if name.endswith((".fa", ".fasta", ".faa")):
        # The FASTA reader needs a path, so the upload is spooled to a temp file.
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / upload.name
        tmp.write_bytes(upload.getvalue())
        return tmp
    if name.endswith((".tsv", ".tab")):
        return pd.read_csv(io.BytesIO(upload.getvalue()), sep="\t")
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(upload.getvalue()))
    return pd.read_csv(io.BytesIO(upload.getvalue()))


def _show_report(report, title: str) -> None:
    st.markdown(f"**{title}**")
    st.write(f"{report.n_rows_kept} of {report.n_rows_read} rows usable.")
    counts = report.counts_by_code()
    if counts:
        st.dataframe(
            pd.DataFrame(
                [{"reason": code, "rows": count} for code, count in counts.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("Every excluded row, with its reason"):
            st.dataframe(exclusions_frame(report.exclusions), hide_index=True)
    for note in report.notes:
        st.caption(note)
    if report.has_blocking:
        st.error(
            f"{len(report.blocking)} row(s) cannot be used as given. Fix the input file, or"
            " continue knowing those rows are excluded. AssayPilot will not alter mutation"
            " numbering or drop values to make them fit."
        )


# -------------------------------------------------------------- 1. campaign


def screen_campaign() -> None:
    st.header("1 · Campaign")
    st.caption(
        "A campaign is one protein, one quantitative assay, one experimental condition."
        " Measurements from another condition belong to a different campaign and are never"
        " pooled automatically."
    )

    existing = st.file_uploader(
        "Load an existing campaign.json", type=["json"], key="campaign_file"
    )
    if existing is not None:
        import json
        import tempfile

        path = Path(tempfile.mkdtemp()) / "campaign.json"
        path.write_bytes(existing.getvalue())
        try:
            st.session_state["campaign"] = Campaign.from_dict(json.loads(path.read_text()))
            st.success(f"Loaded campaign {st.session_state['campaign'].campaign_id}")
        except AssayPilotError as exc:
            st.error(str(exc))

    campaign: Campaign | None = st.session_state.get("campaign")

    with st.form("campaign_form"):
        columns = st.columns(2)
        campaign_id = columns[0].text_input(
            "Campaign id", value=campaign.campaign_id if campaign else "my-campaign"
        )
        condition_id = columns[1].text_input(
            "Condition label",
            value=campaign.condition_id if campaign else "37C_pH7.4",
            help="For example 37C_pH7.4_lysate. Changing the condition means a new campaign.",
        )
        reference = st.text_area(
            "Reference sequence",
            value=campaign.reference_sequence if campaign else "",
            height=120,
            help="Standard amino acids only. Variants must be substitutions of this sequence.",
        )
        columns = st.columns(3)
        assay_name = columns[0].text_input(
            "Assay name", value=campaign.assay_name if campaign else "initial rate"
        )
        assay_units = columns[1].text_input(
            "Units", value=campaign.assay_units if campaign else "s^-1"
        )
        objective = columns[2].selectbox(
            "Objective",
            ["maximize", "minimize"],
            index=0 if not campaign or campaign.objective == "maximize" else 1,
        )
        columns = st.columns(2)
        use_threshold = columns[0].checkbox(
            "Define a success threshold",
            value=bool(campaign and campaign.success_threshold is not None),
            help=(
                "A hit rate can only be reported against a threshold defined before the"
                " results are seen."
            ),
        )
        threshold = columns[1].number_input(
            "Threshold value",
            value=(
                float(campaign.success_threshold)
                if campaign and campaign.success_threshold
                else 0.0
            ),
        )
        batch_size = st.number_input(
            "Batch size (candidate wells)",
            min_value=1,
            value=int(campaign.selection.batch_size) if campaign else 24,
        )
        submitted = st.form_submit_button("Save campaign")

    if submitted:
        try:
            st.session_state["campaign"] = Campaign(
                campaign_id=campaign_id,
                reference_sequence=reference,
                assay_name=assay_name,
                assay_units=assay_units,
                objective=objective,
                condition_id=condition_id,
                success_threshold=float(threshold) if use_threshold else None,
                selection=SelectionSettings(batch_size=int(batch_size)),
            )
            st.success("Campaign saved. Move on to Data review.")
        except AssayPilotError as exc:
            st.error(str(exc))

    campaign = st.session_state.get("campaign")
    if campaign:
        st.download_button(
            "Download campaign.json",
            data=pd.Series(campaign.to_dict()).to_json(indent=2),
            file_name="campaign.json",
            mime="application/json",
        )
        if campaign.success_threshold is None:
            st.info(
                "No success threshold is defined, so hit rates will not be reported. Rank"
                " correlation will be used as the primary metric instead."
            )


# ----------------------------------------------------------- 2. data review


def screen_data() -> None:
    st.header("2 · Data review")
    campaign: Campaign | None = st.session_state.get("campaign")
    if campaign is None:
        st.warning("Define a campaign first.")
        return

    columns = st.columns(2)
    measurements_upload = columns[0].file_uploader(
        "Measurements", type=["csv", "tsv", "parquet"], key="measurements_file"
    )
    candidates_upload = columns[1].file_uploader(
        "Candidates", type=["csv", "tsv", "parquet", "fasta", "fa"], key="candidates_file"
    )
    transform = st.selectbox(
        "Value transformation",
        ["none", "log10", "log2", "ln", "zscore"],
        help="Raw values are always kept alongside any transformation.",
    )

    if measurements_upload is not None:
        try:
            measurements = load_measurements(
                _read_upload(measurements_upload), campaign, strict=False
            )
            if transform != "none":
                measurements = apply_transform(measurements, transform)
            st.session_state["measurements"] = measurements
        except AssayPilotError as exc:
            st.error(str(exc))

    if candidates_upload is not None:
        try:
            st.session_state["candidates"] = load_candidates(
                _read_upload(candidates_upload), campaign, strict=False
            )
        except AssayPilotError as exc:
            st.error(str(exc))

    measurements = st.session_state.get("measurements")
    candidates = st.session_state.get("candidates")

    if measurements is not None:
        _show_report(measurements.report, "Measurements")
        columns = st.columns(4)
        columns[0].metric("Usable rows", len(measurements))
        columns[1].metric("Distinct sequences", measurements.n_unique_sequences)
        columns[2].metric("Rounds", len(measurements.rounds()) or "none")
        columns[3].metric("Transform", measurements.transform)

        conflicts = measurements.conflicting_replicates()
        if not conflicts.empty:
            st.warning(
                f"{len(conflicts)} sequence(s) have replicates disagreeing by more than half"
                " their mean. They are shown here, not resolved: all raw values stay in the"
                " data."
            )
            st.dataframe(conflicts, hide_index=True, use_container_width=True)

        with st.expander("Evaluation protocol this data supports"):
            try:
                plan = choose_split_plan(measurements, candidates)
                st.write(plan.rationale)
                st.caption(f"This is an {plan.label}.")
                st.dataframe(describe_splits(plan.splits), hide_index=True)
                for warning in plan.warnings:
                    st.warning(warning)
            except AssayPilotError as exc:
                st.error(str(exc))

    if candidates is not None:
        _show_report(candidates.report, "Candidates")
        st.metric("Usable candidates", len(candidates))


# --------------------------------------------------------- 3 + 4. the plan


def _suite_controls() -> SuiteConfig:
    st.subheader("Representation")
    available = esm_available()
    if not available:
        st.info(
            "torch and transformers are not installed, so ESM-2 is unavailable. The"
            " physicochemical representation is used instead — it is not a protein language"
            " model, and results are labelled accordingly. Install with"
            " `pip install \"assaypilot[esm]\"` to enable ESM-2."
        )
    representation = st.selectbox(
        "Sequence representation",
        ["esm2", "physchem"] if available else ["physchem"],
        help="ESM-2 embeddings, or a dependency-free physicochemical representation.",
    )
    checkpoint = DEFAULT_CHECKPOINT
    revision = None
    if representation == "esm2":
        checkpoint = st.text_input("ESM-2 checkpoint", value=DEFAULT_CHECKPOINT)
        revision = st.text_input(
            "Checkpoint revision (recommended)",
            value="",
            help=(
                "A model name alone is not enough to reproduce a result. Without a revision"
                " the manifest records 'unpinned'."
            ),
        )
    return SuiteConfig(
        representation=representation,
        esm_checkpoint=checkpoint,
        esm_revision=revision or None,
    )


def screen_plan() -> None:
    st.header("3 · Evidence and batch")
    campaign: Campaign | None = st.session_state.get("campaign")
    measurements = st.session_state.get("measurements")
    candidates = st.session_state.get("candidates")

    if campaign is None or candidates is None:
        st.warning("Load a campaign and a candidate pool first.")
        return

    with st.sidebar:
        st.subheader("Selection settings")
        batch_size = st.number_input(
            "Batch size", min_value=1, value=int(campaign.selection.batch_size)
        )
        policy = st.radio(
            "Policy",
            ["top_ranked", "diversity"],
            help=(
                "Diversity suppresses candidates closer than a visible Hamming distance to"
                " one already chosen. It is an option to test, not an assumed improvement."
            ),
        )
        min_distance = (
            st.number_input("Minimum distance between picks", min_value=1, value=2)
            if policy == "diversity"
            else None
        )
        max_substitutions = st.number_input(
            "Maximum substitutions (0 = no limit)", min_value=0, value=0
        )
        locked_text = st.text_input("Locked positions (comma separated)", value="")
        replicates = st.number_input("Replicates per candidate", min_value=1, value=1)
        controls_text = st.text_input("Control well labels (comma separated)", value="")
        restrict = st.checkbox(
            "Restrict to the evaluated region",
            help=(
                "Exclude candidates further from the measured variants than the evaluation"
                " covered."
            ),
        )
        refits = st.number_input(
            "Bootstrap refits for ranking stability", min_value=0, value=0, step=5
        )
        suite = _suite_controls()

    locked = tuple(int(p) for p in locked_text.replace(" ", "").split(",") if p)
    controls = tuple(c.strip() for c in controls_text.split(",") if c.strip())
    settings = SelectionSettings(
        batch_size=int(batch_size),
        locked_positions=locked,
        max_substitutions=int(max_substitutions) or None,
        diversity_min_distance=int(min_distance) if min_distance else None,
        control_sequences=controls,
        replicates_per_candidate=int(replicates),
    )

    if st.button("Plan the next batch", type="primary"):
        with st.spinner("Comparing methods and selecting a batch…"):
            try:
                st.session_state["plan"] = plan_batch(
                    campaign,
                    candidates,
                    measurements,
                    settings=settings,
                    suite=suite,
                    policy=policy,
                    restrict_to_evaluated_range=restrict,
                    rank_variability_refits=int(refits),
                )
            except AssayPilotError as exc:
                st.error(str(exc))

    plan = st.session_state.get("plan")
    if plan is None:
        return

    icon, label = EVIDENCE_STYLE[plan.evidence_level]
    (st.success if plan.evidence_level == "supported" else st.warning)(f"{icon} {label}")
    st.write(plan.headline())

    budget = plan.budget
    columns = st.columns(4)
    columns[0].metric("Candidates selected", budget["candidates_selected"])
    columns[1].metric("Candidate wells", budget["candidate_wells"])
    columns[2].metric("Control wells", budget["control_wells"])
    columns[3].metric("Total wells", budget["total_wells"])

    evidence_tab, batch_tab, coverage_tab, export_tab = st.tabs(
        ["Model evidence", "The batch", "Coverage", "Export"]
    )

    with evidence_tab:
        if plan.evaluation is None:
            st.info(
                "No held-out evaluation was possible, so there is no assay-specific evidence"
                " for any method."
            )
        else:
            evaluation = plan.evaluation
            st.markdown("**Prediction task**")
            for split in evaluation.split_plan.splits:
                st.write(f"- {split.description}")
            st.caption(
                f"{evaluation.split_plan.rationale}"
                f" This is an {evaluation.split_plan.label}."
            )

            st.markdown("**How each method scored**")
            comparison = method_comparison_frame(plan)
            st.dataframe(
                comparison[comparison["scope"] == "aggregate"][
                    ["method_name", "spearman", "hit_rate_at_k", "best_at_k", "mean_at_k"]
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                f"Primary metric: {HUMAN_NAMES.get(evaluation.primary_metric)}."
                f" Random selection of {evaluation.k} variants from the same held-out set"
                f" would have reached a best value of"
                f" {evaluation.random_reference.get('best_at_k'):.3f}."
            )
            with st.expander("Per-split detail"):
                st.dataframe(comparison, hide_index=True, use_container_width=True)

            st.markdown("**Paired comparisons**")
            for comparison_row in evaluation.comparisons:
                st.write(f"- {comparison_row.sentence}")

            st.markdown("**The rule that chose the method**")
            st.caption(evaluation.recommendation.rule)
            for reason in evaluation.recommendation.reasons:
                st.write(f"- {reason}")

            if evaluation.unavailable:
                st.markdown("**Methods that could not be run**")
                for key, reason in evaluation.unavailable.items():
                    st.write(f"- `{key}`: {reason}")

        if plan.rank_variability is not None:
            st.markdown("**Ranking stability**")
            st.caption(RANK_VARIABILITY_CAVEAT)
            st.dataframe(
                plan.rank_variability.frame.head(25), hide_index=True, use_container_width=True
            )

    with batch_tab:
        columns = [
            c
            for c in (
                "selection_rank",
                "design_id",
                "candidate_id",
                "mutations",
                "score",
                "predicted_value",
                "min_distance_to_measured",
                "within_evaluated_range",
                "top_k_frequency",
                "selection_reason",
            )
            if c in plan.selected.columns
        ]
        st.dataframe(plan.selected[columns], hide_index=True, use_container_width=True)
        if plan.outcome.shortfall:
            st.warning(
                f"The batch is short by {plan.outcome.shortfall}: {plan.outcome.shortfall_reason}"
            )
        st.caption(
            "design_id is immutable: the same sequence in this campaign always receives the"
            " same identifier, so results can be matched back to this plan."
        )

    with coverage_tab:
        st.dataframe(plan.coverage.frame, hide_index=True, use_container_width=True)
        for warning in plan.coverage.warnings:
            st.warning(warning)

    with export_tab:
        output = st.text_input("Output directory", value=f"plan_{plan.run_id}")
        if st.button("Write exports"):
            written = export_plan(plan, output)
            st.success("Wrote: " + ", ".join(str(p) for p in written.values()))
        st.download_button(
            "Download selected_variants.csv",
            data=plan.selected.to_csv(index=False),
            file_name="selected_variants.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download run_manifest.json",
            data=plan.manifest.to_json(),
            file_name="run_manifest.json",
            mime="application/json",
        )
        st.download_button(
            "Download report.html",
            data=render_report(plan),
            file_name="report.html",
            mime="text/html",
        )


# -------------------------------------------------------------- 4. rounds


def screen_rounds() -> None:
    st.header("4 · Next round")
    plan = st.session_state.get("plan")
    campaign: Campaign | None = st.session_state.get("campaign")
    if plan is None or campaign is None:
        st.warning("Produce a plan first, then come back when the results arrive.")
        return

    st.caption(
        "Predictions are scored against the new measurements before anything is retrained."
        " This is the only prospective evidence AssayPilot ever produces."
    )
    upload = st.file_uploader("New round measurements", type=["csv", "tsv", "parquet"])
    if upload is None:
        return

    try:
        results = load_measurements(_read_upload(upload), campaign, strict=False)
    except AssayPilotError as exc:
        st.error(str(exc))
        return

    _show_report(results.report, "New measurements")
    scorecard = score_round(
        plan.manifest,
        results,
        campaign=campaign,
        previous_measurements=st.session_state.get("measurements"),
    )

    for line in scorecard.summary_lines():
        st.write(f"- {line}")
    if not scorecard.matched.empty:
        st.dataframe(scorecard.matched, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------- layout


def main() -> None:
    st.title("AssayPilot")
    st.caption(
        "Plan the next round of protein variant experiments from the measurements you already"
        " have. Everything runs locally; no measurement leaves this machine."
    )

    campaign = st.session_state.get("campaign")
    if campaign is not None:
        st.sidebar.success(
            f"**{campaign.campaign_id}**\n\n{campaign.assay_name} ({campaign.assay_units})\n\n"
            f"condition `{campaign.condition_id}` · {campaign.objective}"
        )

    tabs = st.tabs(["1 · Campaign", "2 · Data review", "3 · Evidence and batch", "4 · Next round"])
    with tabs[0]:
        screen_campaign()
    with tabs[1]:
        screen_data()
    with tabs[2]:
        screen_plan()
    with tabs[3]:
        screen_rounds()


main()

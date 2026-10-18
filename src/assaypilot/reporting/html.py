"""The human-readable HTML report.

The report is written for the researcher who has to defend the batch at a lab
meeting. It states, in order: what was decided, on what evidence, what the
evidence does not cover, what was excluded and why, and how to reproduce the
run.

It is a single self-contained file with no external assets, so it can be
emailed, attached to an electronic lab notebook, or archived next to the
plate map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from jinja2 import Environment, select_autoescape
from markupsafe import Markup

from ..evaluation.metrics import HUMAN_NAMES
from ..evaluation.uncertainty import RANK_VARIABILITY_CAVEAT

if TYPE_CHECKING:  # avoids a circular import with the pipeline module
    from ..pipeline import ExperimentPlan

#: Mirrors assaypilot.pipeline, imported lazily to keep this module import-light.
EVIDENCE_SUPPORTED = "supported"
EVIDENCE_EXPLORATORY = "exploratory"
EVIDENCE_NONE = "none"

_ENV = Environment(autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)

EVIDENCE_BANNERS: dict[str, tuple[str, str]] = {
    EVIDENCE_SUPPORTED: (
        "Supported by held-out evaluation",
        "The chosen method was compared against the others on held-out measurements from this"
        " campaign, and the comparison supports the choice.",
    ),
    EVIDENCE_EXPLORATORY: (
        "Exploratory — evidence does not support a reliable ordering of methods",
        "The held-out evaluation was too small, too unstable, or too narrow to separate the"
        " methods. The batch is a reasonable next step, not a validated recommendation.",
    ),
    EVIDENCE_NONE: (
        "Exploratory — no assay-specific validation",
        "No measured data from this campaign was available, so no method could be validated"
        " on this assay. The ranking comes from a pretrained model that has never seen these"
        " measurements.",
    ),
}

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AssayPilot plan — {{ campaign.campaign_id }} — {{ run_id }}</title>
<style>
  :root {
    --bg: #ffffff; --fg: #15181d; --muted: #5b6471; --line: #d9dee5;
    --panel: #f6f8fa; --accent: #1f5f8b; --warn-bg: #fff6e5; --warn-line: #e0a528;
    --ok-bg: #eef7ef; --ok-line: #3f8f54;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171c; --fg: #e8ecf1; --muted: #9aa5b4; --line: #2c333d;
      --panel: #1b1f26; --accent: #76b2dc; --warn-bg: #2f2718; --warn-line: #d7a33c;
      --ok-bg: #18271c; --ok-line: #5fa873;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0 auto; padding: 32px 20px 72px; max-width: 1040px;
    background: var(--bg); color: var(--fg);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  h1 { font-size: 26px; margin: 0 0 4px; }
  h2 { font-size: 19px; margin: 36px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
  h3 { font-size: 15px; margin: 22px 0 8px; }
  p, li { margin: 8px 0; }
  .sub { color: var(--muted); margin: 0 0 22px; }
  .banner { padding: 14px 16px; border-radius: 8px; border-left: 4px solid; margin: 18px 0; }
  .banner.ok { background: var(--ok-bg); border-color: var(--ok-line); }
  .banner.warn { background: var(--warn-bg); border-color: var(--warn-line); }
  .banner strong { display: block; margin-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; white-space: nowrap; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
  .muted { color: var(--muted); }
  .small { font-size: 13px; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
  ul.reasons { padding-left: 20px; }
  .scroll { overflow-x: auto; }
  footer { margin-top: 48px; color: var(--muted); font-size: 13px; border-top: 1px solid var(--line); padding-top: 14px; }
</style>
</head>
<body>

<h1>Next-round plan — {{ campaign.assay_name }}</h1>
<p class="sub">
  Campaign <code>{{ campaign.campaign_id }}</code> · condition <code>{{ campaign.condition_id }}</code>
  · objective {{ campaign.objective }} {{ campaign.assay_units }}
  · run <code>{{ run_id }}</code> · {{ created_at }}
</p>

<div class="banner {{ 'ok' if evidence_level == 'supported' else 'warn' }}">
  <strong>{{ evidence_title }}</strong>
  {{ evidence_detail }}
</div>

<p>{{ headline }}</p>

<h2>1. The batch</h2>
<div class="panel small">
  <strong>Budget.</strong> {{ budget.candidates_selected }} candidate(s) selected of
  {{ budget.candidates_requested }} requested, {{ budget.replicates_per_candidate }} replicate(s)
  each = {{ budget.candidate_wells }} candidate well(s), plus {{ budget.control_wells }} control
  well(s) = <strong>{{ budget.total_wells }} well(s) in total</strong>. Candidate count and
  control wells are budgeted separately and never merged.
</div>
{% if shortfall %}
<div class="banner warn">
  <strong>The requested batch could not be filled.</strong>
  {{ shortfall_reason }} The constraint was not relaxed to reach the requested count.
</div>
{% endif %}
<div class="scroll">
{{ selected_table }}
</div>
<p class="small muted">
  <code>score</code> is the ranking value produced by the method
  ({{ method_name }}).
  {% if has_predicted_values %}
  <code>predicted_value</code> is that score expressed in {{ campaign.assay_units }}.
  {% else %}
  This method produces a score on an arbitrary scale, not a predicted assay value, so
  <code>predicted_value</code> is empty.
  {% endif %}
  <code>observed_value</code> is filled only where the sequence has already been measured.
  <code>design_id</code> is immutable: the same sequence in this campaign always receives the
  same identifier, so results can be matched back to this plan.
</p>

<h2>2. Model evidence</h2>
{% if evaluation %}
<p><strong>Prediction task.</strong> {{ split_description }}</p>
<p class="small muted">{{ split_rationale }} This is an <strong>{{ split_label }}</strong>.</p>

<h3>How each method scored</h3>
<div class="scroll">
{{ comparison_table }}
</div>
<p class="small muted">
  Primary metric for the decision: {{ primary_metric_name }}.
  Values are means across {{ n_splits }} split(s); the ± figure is the standard deviation across
  splits, not a confidence interval. Random selection of the same number of variants from the
  same held-out set would have reached a best value of
  {{ random_best }} and a hit rate of {{ random_hit_rate }}.
</p>

<h3>Paired comparisons</h3>
<ul class="reasons small">
{% for sentence in comparison_sentences %}
  <li>{{ sentence }}</li>
{% endfor %}
</ul>
<p class="small muted">
  Intervals are 95% percentile intervals from {{ n_bootstrap }} paired bootstrap resamples over
  distinct held-out sequences. Replicates of a sequence are always resampled together.
</p>

<h3>The rule that chose the method</h3>
<p class="small">{{ recommendation_rule }}</p>
<ul class="reasons small">
{% for reason in recommendation_reasons %}
  <li>{{ reason }}</li>
{% endfor %}
</ul>
{% else %}
<p>
  No held-out evaluation was performed for this run, so there is no assay-specific evidence
  for any method. {{ no_evaluation_reason }}
</p>
{% endif %}

{% if unavailable %}
<h3>Methods that could not be run</h3>
<ul class="reasons small">
{% for key, reason in unavailable.items() %}
  <li><code>{{ key }}</code>: {{ reason }}</li>
{% endfor %}
</ul>
{% endif %}

<h3>What this method cannot do</h3>
<ul class="reasons small">
{% for caveat in method_caveats %}
  <li>{{ caveat }}</li>
{% endfor %}
</ul>

{% if rank_variability %}
<h3>Ranking stability</h3>
<p class="small">{{ rank_variability_caveat }}</p>
<div class="scroll">
{{ rank_variability_table }}
</div>
{% endif %}

<h2>3. How well the evidence covers these candidates</h2>
<div class="scroll">
{{ coverage_table }}
</div>
{% if coverage_warnings %}
<ul class="reasons small">
{% for warning in coverage_warnings %}
  <li>{{ warning }}</li>
{% endfor %}
</ul>
{% endif %}

<h2>4. Data and exclusions</h2>
<div class="scroll">
{{ data_table }}
</div>
{% if exclusion_counts %}
<h3>Why rows and candidates were set aside</h3>
<div class="scroll">
{{ exclusion_table }}
</div>
<p class="small muted">
  Every excluded row is listed individually in <code>exclusions.csv</code>. Nothing was dropped
  without a recorded reason, and no mutation numbering was altered to make an input fit.
</p>
{% endif %}
{% if replicate_conflicts is not none %}
<h3>Replicates that disagree</h3>
<div class="scroll">
{{ replicate_conflicts }}
</div>
<p class="small muted">
  These are shown, not resolved. All raw values remain in the data.
</p>
{% endif %}

{% if warnings %}
<h2>5. Warnings</h2>
<ul class="reasons">
{% for warning in warnings %}
  <li>{{ warning }}</li>
{% endfor %}
</ul>
{% endif %}

<h2>{{ '6' if warnings else '5' }}. Reproducing this run</h2>
<div class="scroll">
{{ repro_table }}
</div>
<p class="small muted">
  The full record — input checksums, partition assignments, seeds, hyperparameters chosen,
  every prediction, every exclusion — is in <code>run_manifest.json</code>.
</p>

<footer>
  Generated by AssayPilot {{ assaypilot_version }}. This report describes a retrospective and
  prospective modelling exercise on measurements supplied by the user. It is not a claim about
  any variant's behaviour beyond the assay and condition named above.
</footer>

</body>
</html>
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, *, max_rows: int | None = None, index: bool = False) -> Markup:
    """Render a frame as an HTML table.

    The result is marked safe because pandas escapes cell contents itself; the
    surrounding template keeps autoescaping on for every other value.
    """
    if frame is None or frame.empty:
        return Markup('<p class="muted small">Nothing to show.</p>')
    shown = frame.head(max_rows) if max_rows else frame
    html = shown.to_html(
        index=index,
        na_rep="—",
        float_format=lambda v: f"{v:.3f}",
        border=0,
        justify="left",
    )
    if max_rows and len(frame) > max_rows:
        html += (
            f'<p class="muted small">Showing {max_rows} of {len(frame)} rows;'
            " the full table is in the exported CSV.</p>"
        )
    return Markup(html)


def _comparison_table(plan: ExperimentPlan) -> pd.DataFrame:
    evaluation = plan.evaluation
    assert evaluation is not None
    rows: list[dict[str, Any]] = []
    for key, summary in evaluation.summaries.items():
        row: dict[str, Any] = {
            "method": summary.info.name,
            "uses labels": "yes" if summary.info.uses_labels else "no",
        }
        for metric in ("spearman", "hit_rate_at_k", "best_at_k", "mean_at_k"):
            mean = summary.mean(metric)
            spread = summary.spread_metrics.get(metric, float("nan"))
            if np.isfinite(mean) and np.isfinite(spread):
                row[HUMAN_NAMES[metric]] = f"{mean:.3f} ± {spread:.3f}"
            elif np.isfinite(mean):
                row[HUMAN_NAMES[metric]] = f"{mean:.3f}"
            else:
                row[HUMAN_NAMES[metric]] = "—"
        row["chosen"] = "yes" if key == evaluation.recommendation.method_key else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_table(plan: ExperimentPlan) -> pd.DataFrame:
    summary = plan.coverage.summary
    rows = [
        ("Candidates in the pool", summary.get("n_candidates")),
        (
            "Median substitutions from the nearest measured variant",
            summary.get("candidate_distance_median"),
        ),
        ("Furthest candidate from any measured variant", summary.get("candidate_distance_max")),
        (
            "Furthest held-out variant the evaluation covered",
            summary.get("max_evaluated_distance"),
        ),
        (
            "Candidates beyond the evaluated region",
            summary.get("n_outside_evaluated_range"),
        ),
        (
            "Candidates carrying a substitution never measured",
            (
                f"{summary['fraction_with_unseen_substitutions']:.0%}"
                if summary.get("fraction_with_unseen_substitutions") is not None
                else None
            ),
        ),
    ]
    return pd.DataFrame([{"quantity": name, "value": _fmt(value, 1)} for name, value in rows])


def _data_table(plan: ExperimentPlan) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if plan.measurements is not None:
        report = plan.measurements.report
        rows.extend(
            [
                {"input": "measurements", "quantity": "rows read", "value": report.n_rows_read},
                {"input": "measurements", "quantity": "rows usable", "value": report.n_rows_kept},
                {
                    "input": "measurements",
                    "quantity": "distinct sequences",
                    "value": plan.measurements.n_unique_sequences,
                },
                {
                    "input": "measurements",
                    "quantity": "rounds present",
                    "value": ", ".join(str(r) for r in plan.measurements.rounds()) or "none",
                },
                {
                    "input": "measurements",
                    "quantity": "value transform",
                    "value": plan.measurements.transform,
                },
            ]
        )
    else:
        rows.append({"input": "measurements", "quantity": "supplied", "value": "none"})

    candidate_report = plan.candidates.report
    rows.extend(
        [
            {"input": "candidates", "quantity": "rows read", "value": candidate_report.n_rows_read},
            {
                "input": "candidates",
                "quantity": "rows usable",
                "value": candidate_report.n_rows_kept,
            },
            {
                "input": "candidates",
                "quantity": "eligible after filtering",
                "value": plan.filter_result.n_eligible,
            },
        ]
    )
    return pd.DataFrame(rows)


def _exclusion_table(plan: ExperimentPlan) -> pd.DataFrame:
    from ..data.validation import ExclusionCode

    counts: dict[str, int] = {}
    for exclusion in plan.filter_result.exclusions:
        counts[exclusion.code] = counts.get(exclusion.code, 0) + 1
    for source in (plan.candidates.report, plan.measurements.report if plan.measurements else None):
        if source is None:
            continue
        for code, count in source.counts_by_code().items():
            counts[code] = counts.get(code, 0) + count

    rows = [
        {
            "reason": code,
            "count": count,
            "meaning": ExclusionCode.DESCRIPTIONS.get(code, "restricted by the researcher"),
        }
        for code, count in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return pd.DataFrame(rows)


def _repro_table(plan: ExperimentPlan) -> pd.DataFrame:
    manifest = plan.manifest
    config = manifest.config
    suite = config.get("suite", {})
    environment = manifest.environment
    rows = [
        ("run id", manifest.run_id),
        ("manifest fingerprint", manifest.fingerprint()[:16]),
        ("random seed", config.get("seed")),
        ("bootstrap resamples", config.get("n_bootstrap")),
        ("selection policy", config.get("selection_policy")),
        ("split policy", config.get("split_policy")),
        ("representation", suite.get("representation")),
        ("ESM checkpoint", suite.get("esm_checkpoint")),
        ("ESM revision", suite.get("esm_revision") or "unpinned"),
        ("chosen hyperparameters", _short_dict(config.get("fit_details", {}))),
        ("AssayPilot", manifest.assaypilot_version),
        ("Python", environment.get("python")),
        (
            "key packages",
            ", ".join(
                f"{name} {version}"
                for name, version in environment.get("packages", {}).items()
                if version != "not installed"
            ),
        ),
    ]
    return pd.DataFrame([{"setting": name, "value": str(value)} for name, value in rows])


def _short_dict(data: dict[str, Any], limit: int = 4) -> str:
    items = [
        f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list)) and v is not None
    ][:limit]
    return ", ".join(items) or "—"


def build_context(plan: ExperimentPlan) -> dict[str, Any]:
    """Assemble the template context for a plan (useful for tests and custom templates)."""
    title, detail = EVIDENCE_BANNERS[plan.evidence_level]
    evaluation = plan.evaluation

    selected_display_columns = [
        c
        for c in (
            "selection_rank",
            "design_id",
            "candidate_id",
            "mutations",
            "n_substitutions",
            "score",
            "predicted_value",
            "observed_value",
            "min_distance_to_measured",
            "within_evaluated_range",
            "top_k_frequency",
            "selection_reason",
        )
        if c in plan.selected.columns
    ]

    context: dict[str, Any] = {
        "campaign": plan.campaign.to_dict(),
        "run_id": plan.run_id,
        "created_at": plan.manifest.created_at,
        "assaypilot_version": plan.manifest.assaypilot_version,
        "evidence_level": plan.evidence_level,
        "evidence_title": title,
        "evidence_detail": detail,
        "headline": plan.headline(),
        "budget": plan.budget,
        "shortfall": plan.outcome.shortfall,
        "shortfall_reason": plan.outcome.shortfall_reason,
        "selected_table": _table(plan.selected[selected_display_columns]),
        "method_name": plan.method_name,
        "has_predicted_values": bool(plan.selected["predicted_value"].notna().any())
        if "predicted_value" in plan.selected
        else False,
        "method_caveats": list(
            next(
                (
                    summary.info.caveats
                    for key, summary in (evaluation.summaries.items() if evaluation else [])
                    if key == plan.method_key
                ),
                (),
            )
        )
        or ["No method-specific caveats were recorded for this method."],
        "coverage_table": _table(_coverage_table(plan)),
        "coverage_warnings": plan.coverage.warnings,
        "data_table": _table(_data_table(plan)),
        "exclusion_counts": True,
        "exclusion_table": _table(_exclusion_table(plan)),
        "replicate_conflicts": None,
        "warnings": plan.warnings,
        "repro_table": _table(_repro_table(plan)),
        "unavailable": evaluation.unavailable if evaluation else {},
        "no_evaluation_reason": (
            "This happens when no measurements were supplied, or when there were too few"
            " measured sequences to build any held-out split."
        ),
        "rank_variability": plan.rank_variability is not None,
        "rank_variability_caveat": RANK_VARIABILITY_CAVEAT,
        "rank_variability_table": (
            _table(plan.rank_variability.frame.head(25)) if plan.rank_variability else ""
        ),
    }

    exclusion_frame = _exclusion_table(plan)
    context["exclusion_counts"] = not exclusion_frame.empty

    if plan.measurements is not None:
        conflicts = plan.measurements.conflicting_replicates()
        if not conflicts.empty:
            context["replicate_conflicts"] = _table(
                conflicts[
                    ["variant_id", "mutations", "n_measurements", "mean_value", "range_value"]
                ],
                max_rows=15,
            )

    if evaluation is not None:
        context.update(
            {
                "evaluation": True,
                "split_description": "; ".join(s.description for s in evaluation.split_plan.splits),
                "split_rationale": evaluation.split_plan.rationale,
                "split_label": evaluation.split_plan.label,
                "n_splits": len(evaluation.split_plan.splits),
                "comparison_table": _table(_comparison_table(plan)),
                "comparison_sentences": [c.sentence for c in evaluation.comparisons],
                "primary_metric_name": HUMAN_NAMES.get(
                    evaluation.primary_metric, evaluation.primary_metric
                ),
                "random_best": _fmt(evaluation.random_reference.get("best_at_k")),
                "random_hit_rate": _fmt(evaluation.random_reference.get("hit_rate_at_k")),
                "recommendation_rule": evaluation.recommendation.rule,
                "recommendation_reasons": evaluation.recommendation.reasons,
                "n_bootstrap": evaluation.n_bootstrap,
            }
        )
    else:
        context["evaluation"] = False

    return context


def render_report(plan: ExperimentPlan) -> str:
    """Render the self-contained HTML report for a plan."""
    template = _ENV.from_string(_TEMPLATE)
    return template.render(**build_context(plan))

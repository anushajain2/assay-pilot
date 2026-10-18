"""The ``assaypilot`` command line.

The CLI and the Streamlit interface share one implementation of the decision
(:mod:`assaypilot.pipeline`), so a plan made at the command line and a plan
made in the browser are the same plan, with the same manifest.

Run ``assaypilot --help`` for the command list, or ``assaypilot demo`` to see
the whole workflow on generated data without downloading anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .campaign import Campaign, SelectionSettings
from .data.loaders import apply_transform, load_candidates, load_measurements
from .errors import AssayPilotError
from .evaluation.splits import choose_split_plan, describe_splits
from .models.registry import SuiteConfig
from .pipeline import plan_batch
from .reporting.exports import export_plan, method_comparison_frame
from .reporting.manifest import RunManifest, file_checksum
from .scoring import score_round

LOG_FORMAT = "%(levelname)s: %(message)s"


# --------------------------------------------------------------------- helpers


def _print_report(title: str, lines: Sequence[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for line in lines:
        print(line)


def _suite_from_args(args: argparse.Namespace) -> SuiteConfig:
    return SuiteConfig(
        representation=args.representation,
        esm_checkpoint=args.esm_checkpoint,
        esm_revision=args.esm_revision,
        device=args.device,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )


def _settings_from_args(campaign: Campaign, args: argparse.Namespace) -> SelectionSettings:
    base = campaign.selection
    locked = tuple(int(p) for p in (args.lock_position or [])) or base.locked_positions
    return SelectionSettings(
        batch_size=args.batch_size if args.batch_size is not None else base.batch_size,
        locked_positions=locked,
        max_substitutions=(
            args.max_substitutions if args.max_substitutions is not None else base.max_substitutions
        ),
        allow_measured_repeats=args.allow_repeats or base.allow_measured_repeats,
        diversity_min_distance=(
            args.min_distance if args.min_distance is not None else base.diversity_min_distance
        ),
        control_sequences=tuple(args.control or ()) or base.control_sequences,
        replicates_per_candidate=(
            args.replicates if args.replicates is not None else base.replicates_per_candidate
        ),
    )


def _load_inputs(args: argparse.Namespace) -> tuple[Campaign, Any, Any, dict[str, Any]]:
    campaign = Campaign.from_json(args.campaign)
    checksums: dict[str, Any] = {"campaign": file_checksum(args.campaign)}

    measurements = None
    if getattr(args, "measurements", None):
        measurements = load_measurements(args.measurements, campaign, strict=False)
        checksums["measurements"] = file_checksum(args.measurements)
        if getattr(args, "transform", "none") != "none":
            measurements = apply_transform(measurements, args.transform)
        measurements = measurements.for_regression()

    candidates = None
    if getattr(args, "candidates", None):
        candidates = load_candidates(args.candidates, campaign, strict=False)
        checksums["candidates"] = file_checksum(args.candidates)

    return campaign, measurements, candidates, checksums


# -------------------------------------------------------------------- commands


def cmd_init_campaign(args: argparse.Namespace) -> int:
    """Create a campaign definition file."""
    if args.reference_fasta:
        campaign = Campaign.from_fasta(
            args.reference_fasta,
            campaign_id=args.id,
            assay_name=args.assay_name,
            assay_units=args.units,
            objective=args.objective,
            condition_id=args.condition,
            success_threshold=args.threshold,
            description=args.description or "",
            selection=SelectionSettings(batch_size=args.batch_size),
        )
    else:
        campaign = Campaign(
            campaign_id=args.id,
            reference_sequence=args.reference,
            assay_name=args.assay_name,
            assay_units=args.units,
            objective=args.objective,
            condition_id=args.condition,
            success_threshold=args.threshold,
            description=args.description or "",
            selection=SelectionSettings(batch_size=args.batch_size),
        )
    path = campaign.to_json(args.output)
    print(f"Wrote {path}")
    print(
        f"  reference length {campaign.length}, assay {campaign.assay_name!r} in"
        f" {campaign.assay_units}, {campaign.objective}, condition {campaign.condition_id}"
    )
    if campaign.success_threshold is None:
        print(
            "  No success threshold set. Define one before you ask for hit rates, so that the"
            " definition of a hit cannot follow the results."
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Check inputs against the campaign and print what would be used."""
    campaign, measurements, candidates, _ = _load_inputs(args)
    print(
        f"Campaign {campaign.campaign_id}: {campaign.assay_name} ({campaign.assay_units}),"
        f" condition {campaign.condition_id}, reference length {campaign.length}"
    )

    if measurements is not None:
        _print_report("Measurements", measurements.report.summary_lines())
        if measurements.rounds():
            print(f"  rounds: {', '.join(str(r) for r in measurements.rounds())}")
        conflicts = measurements.conflicting_replicates()
        if not conflicts.empty:
            print(
                f"  {len(conflicts)} sequence(s) have replicates disagreeing by more than 50%"
                " of their mean; raw values are kept and nothing was dropped."
            )
        try:
            plan = choose_split_plan(measurements, candidates)
            _print_report("Evaluation protocol that would be used", [plan.rationale])
            print(describe_splits(plan.splits).to_string(index=False))
        except AssayPilotError as exc:
            print(f"\nNo evaluation split could be built: {exc}")

    if candidates is not None:
        _print_report("Candidates", candidates.report.summary_lines())

    blocking = []
    for source in (measurements, candidates):
        if source is not None:
            blocking.extend(source.report.blocking)
    if blocking:
        print(
            f"\n{len(blocking)} row(s) cannot be used as given. They are listed above by reason."
            " Fix the input or rerun planning with --skip-invalid to continue without them."
        )
        return 1
    print("\nNo blocking problems found.")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Produce the next batch, with its evidence and exports."""
    campaign, measurements, candidates, checksums = _load_inputs(args)
    if candidates is None:
        raise AssayPilotError("--candidates is required for planning")

    blocking = list(candidates.report.blocking) + (
        list(measurements.report.blocking) if measurements is not None else []
    )
    if blocking and not args.skip_invalid:
        print(
            f"{len(blocking)} input row(s) cannot be used as given. Run 'assaypilot validate'"
            " to see them, fix the file, or pass --skip-invalid to continue with the"
            " remaining rows.",
            file=sys.stderr,
        )
        return 1

    settings = _settings_from_args(campaign, args)
    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        settings=settings,
        suite=_suite_from_args(args),
        policy=args.policy,
        split_policy=args.split,
        restrict_to_evaluated_range=args.restrict_to_evaluated,
        n_bootstrap=args.bootstrap,
        rank_variability_refits=args.rank_variability,
        seed=args.seed,
        input_checksums=checksums,
    )

    print(plan.headline())
    if plan.evaluation is not None:
        evaluation = plan.evaluation
        _print_report(
            "Prediction task",
            [s.description for s in evaluation.split_plan.splits]
            + [f"({evaluation.split_plan.label})"],
        )
        _print_report(
            "Method comparison",
            [
                method_comparison_frame(plan)
                .query("scope == 'aggregate'")[
                    ["method_name", "spearman", "hit_rate_at_k", "best_at_k"]
                ]
                .to_string(index=False)
            ],
        )
        _print_report("Why this method", evaluation.recommendation.reasons)

    if plan.warnings:
        _print_report("Warnings", [f"- {w}" for w in plan.warnings])

    written = export_plan(plan, args.output)
    _print_report("Exports", [f"{name}: {path}" for name, path in written.items()])

    if args.store:
        from .store import CampaignStore

        with CampaignStore(args.store) as store:
            try:
                store.get_campaign(campaign.campaign_id)
            except AssayPilotError:
                store.create_campaign(campaign)
                if measurements is not None:
                    store.add_round(
                        campaign.campaign_id, measurements, source=str(args.measurements)
                    )
            store.save_run(plan)
        print(f"\nRecorded run {plan.run_id} in {args.store}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Compare the methods without selecting a batch."""
    campaign, measurements, candidates, _ = _load_inputs(args)
    if measurements is None:
        raise AssayPilotError("--measurements is required for evaluation")

    from .evaluation.compare import evaluate_methods
    from .models.registry import build_cache, build_methods, split_by_availability

    suite = _suite_from_args(args)
    cache = build_cache(suite)
    methods, unavailable = split_by_availability(build_methods(campaign, suite, cache=cache))
    for key, reason in unavailable.items():
        print(f"note: {key} unavailable — {reason}")

    split_plan = choose_split_plan(measurements, candidates, policy=args.split, seed=args.seed)
    _print_report("Evaluation protocol", [split_plan.rationale, f"({split_plan.label})"])
    print(describe_splits(split_plan.splits).to_string(index=False))

    result = evaluate_methods(
        methods,
        measurements,
        split_plan,
        campaign=campaign,
        k=args.batch_size or campaign.selection.batch_size,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        unavailable=unavailable,
    )

    rows = []
    for key, summary in result.summaries.items():
        rows.append(
            {
                "method": summary.info.name,
                "spearman": round(summary.mean("spearman"), 3),
                "hit_rate_at_k": round(summary.mean("hit_rate_at_k"), 3),
                "best_at_k": round(summary.mean("best_at_k"), 3),
                "chosen": "yes" if key == result.recommendation.method_key else "",
            }
        )
    import pandas as pd

    _print_report("Results", [pd.DataFrame(rows).to_string(index=False)])
    _print_report("Paired comparisons", [f"- {c.sentence}" for c in result.comparisons])
    _print_report(
        f"Recommendation ({result.recommendation.evidence_level})",
        [result.recommendation.rule] + [f"- {r}" for r in result.recommendation.reasons],
    )

    if args.output:
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


def cmd_score_round(args: argparse.Namespace) -> int:
    """Score a previous plan against the measurements it produced."""
    manifest = RunManifest.read(args.manifest)
    campaign = Campaign.from_dict(manifest.campaign)
    new_measurements = load_measurements(args.measurements, campaign, strict=False)

    previous = None
    if args.previous:
        previous = load_measurements(args.previous, campaign, strict=False)

    scorecard = score_round(
        manifest, new_measurements, campaign=campaign, previous_measurements=previous
    )
    _print_report(f"Scorecard for {scorecard.run_id}", scorecard.summary_lines())
    if not scorecard.matched.empty:
        print()
        print(scorecard.matched.to_string(index=False))

    if args.output:
        Path(args.output).write_text(json.dumps(scorecard.to_dict(), indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")

    if args.store:
        from .store import CampaignStore

        with CampaignStore(args.store) as store:
            store.save_scorecard(scorecard)
            store.add_round(
                campaign.campaign_id,
                new_measurements,
                round_id=args.round_id,
                source=str(args.measurements),
            )
        print(f"Appended the round and scorecard to {args.store}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a simulated campaign and run the whole workflow on it."""
    from .datasets import simulate_campaign

    output = Path(args.output)
    simulated = simulate_campaign(
        n_measured=args.n_measured, n_candidates=args.n_candidates, seed=args.seed
    )
    data_dir = output / "data"
    paths = simulated.write(data_dir)
    print(f"Wrote simulated inputs to {data_dir} (synthetic data — not a real protein)")

    measurements = load_measurements(paths["measurements"], simulated.campaign, strict=False)
    candidates = load_candidates(paths["candidates"], simulated.campaign, strict=False)

    plan = plan_batch(
        simulated.campaign,
        candidates,
        measurements,
        suite=SuiteConfig(representation=args.representation),
        policy=args.policy,
        rank_variability_refits=args.rank_variability,
        seed=args.seed,
        input_checksums={
            "measurements": file_checksum(paths["measurements"]),
            "candidates": file_checksum(paths["candidates"]),
        },
    )
    print()
    print(plan.headline())
    written = export_plan(plan, output / "plan")
    _print_report("Exports", [f"{name}: {path}" for name, path in written.items()])
    print(
        "\nEverything above was produced from synthetic data generated by"
        " assaypilot.datasets.simulate_campaign. It demonstrates the workflow and says"
        " nothing about any real protein."
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Replay a measured dataset round by round and compare strategies."""
    from .benchmarks.replay import ReplayConfig, compare_strategies, replay_campaign

    campaign = Campaign.from_json(args.campaign)
    measurements = load_measurements(args.measurements, campaign, strict=False)

    config = ReplayConfig(
        n_initial=args.n_initial,
        batch_size=args.batch_size or campaign.selection.batch_size,
        n_rounds=args.rounds,
        seeds=tuple(args.seeds),
        suite=_suite_from_args(args),
    )
    result = replay_campaign(campaign, measurements.frame, config)
    _print_report("Replay summary", [result.summary().to_string(index=False)])
    _print_report(
        "Difference from random selection (paired by seed)",
        [compare_strategies(result).to_string(index=False)],
    )
    print(f"\n{result.caveat}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        result.frame.to_csv(out / "replay_rounds.csv", index=False)
        result.summary().to_csv(out / "replay_summary.csv", index=False)
        (out / "replay_config.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"Wrote replay outputs to {out}")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """Launch the Streamlit interface."""
    import subprocess

    app_path = Path(__file__).parent / "app" / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if args.port:
        command += ["--server.port", str(args.port)]
    print("Starting the AssayPilot interface:", " ".join(command))
    try:
        return subprocess.call(command)
    except FileNotFoundError:  # pragma: no cover - depends on the environment
        print(
            "Streamlit is not installed. Install it with 'pip install \"assaypilot[app]\"'.",
            file=sys.stderr,
        )
        return 1


# ---------------------------------------------------------------------- parser


def _add_suite_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("representation")
    group.add_argument(
        "--representation",
        choices=("esm2", "physchem"),
        default="esm2",
        help=(
            "sequence representation for the assay-trained model. 'physchem' needs no model"
            " weights and is labelled as not a protein language model (default: esm2)"
        ),
    )
    group.add_argument("--esm-checkpoint", default="facebook/esm2_t12_35M_UR50D")
    group.add_argument(
        "--esm-revision",
        default=None,
        help="pin the checkpoint revision; without it the manifest records 'unpinned'",
    )
    group.add_argument("--device", default=None, help="torch device (default: auto)")
    group.add_argument("--cache-dir", default=None, help="embedding cache directory")
    group.add_argument("--no-cache", action="store_true", help="do not cache embeddings")


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("selection")
    group.add_argument("--batch-size", type=int, default=None)
    group.add_argument("--policy", choices=("top_ranked", "diversity"), default="top_ranked")
    group.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="minimum Hamming distance between selected candidates (diversity policy)",
    )
    group.add_argument("--lock-position", type=int, action="append", default=None)
    group.add_argument("--max-substitutions", type=int, default=None)
    group.add_argument("--allow-repeats", action="store_true")
    group.add_argument("--control", action="append", default=None)
    group.add_argument("--replicates", type=int, default=None)
    group.add_argument(
        "--restrict-to-evaluated",
        action="store_true",
        help="exclude candidates beyond the region the evaluation covered",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assaypilot",
        description=(
            "Plan the next round of protein variant experiments from the measurements you"
            " already have."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="log more detail")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-campaign
    init = subparsers.add_parser("init-campaign", help="create a campaign definition file")
    init.add_argument("--id", required=True, help="campaign identifier")
    reference = init.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference", help="reference sequence")
    reference.add_argument(
        "--reference-fasta", help="FASTA file whose first record is the reference"
    )
    init.add_argument("--assay-name", required=True, help="what was measured, in your words")
    init.add_argument("--units", required=True, help="units of the measured value")
    init.add_argument("--objective", choices=("maximize", "minimize"), default="maximize")
    init.add_argument("--condition", default="default", help="experimental condition label")
    init.add_argument("--threshold", type=float, default=None, help="success threshold (a hit)")
    init.add_argument("--batch-size", type=int, default=24)
    init.add_argument("--description", default=None)
    init.add_argument("--output", "-o", default="campaign.json")
    init.set_defaults(func=cmd_init_campaign)

    # validate
    validate = subparsers.add_parser("validate", help="check inputs before planning")
    validate.add_argument("--campaign", required=True)
    validate.add_argument("--measurements")
    validate.add_argument("--candidates")
    validate.add_argument(
        "--transform", choices=("none", "log10", "log2", "ln", "zscore"), default="none"
    )
    validate.set_defaults(func=cmd_validate)

    # evaluate
    evaluate = subparsers.add_parser("evaluate", help="compare methods without selecting a batch")
    evaluate.add_argument("--campaign", required=True)
    evaluate.add_argument("--measurements", required=True)
    evaluate.add_argument("--candidates", help="used to aim the evaluation at the right distance")
    evaluate.add_argument(
        "--transform", choices=("none", "log10", "log2", "ln", "zscore"), default="none"
    )
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument(
        "--split",
        choices=("auto", "round", "mutation_distance", "mutation_pattern", "random"),
        default="auto",
    )
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--output", "-o", default=None, help="write the result as JSON")
    _add_suite_arguments(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    # plan
    plan = subparsers.add_parser("plan", help="select the next batch and export it")
    plan.add_argument("--campaign", required=True)
    plan.add_argument("--candidates", required=True)
    plan.add_argument("--measurements", default=None)
    plan.add_argument(
        "--transform", choices=("none", "log10", "log2", "ln", "zscore"), default="none"
    )
    plan.add_argument("--output", "-o", default="plan_output")
    plan.add_argument(
        "--split",
        choices=("auto", "round", "mutation_distance", "mutation_pattern", "random"),
        default="auto",
    )
    plan.add_argument("--bootstrap", type=int, default=500)
    plan.add_argument(
        "--rank-variability",
        type=int,
        default=0,
        metavar="N",
        help="bootstrap refits used to report ranking stability (0 skips it)",
    )
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--store", default=None, help="also record the run in this campaign store")
    plan.add_argument(
        "--skip-invalid",
        action="store_true",
        help="continue when some input rows cannot be used, instead of stopping",
    )
    _add_selection_arguments(plan)
    _add_suite_arguments(plan)
    plan.set_defaults(func=cmd_plan)

    # score-round
    score = subparsers.add_parser(
        "score-round", help="score a previous plan against new measurements"
    )
    score.add_argument("--manifest", required=True, help="run_manifest.json from the plan")
    score.add_argument("--measurements", required=True, help="the new round's measurements")
    score.add_argument("--previous", default=None, help="earlier measurements, for comparison")
    score.add_argument("--round-id", default=None)
    score.add_argument("--store", default=None)
    score.add_argument("--output", "-o", default=None)
    score.set_defaults(func=cmd_score_round)

    # demo
    demo = subparsers.add_parser("demo", help="run the whole workflow on simulated data")
    demo.add_argument("--output", "-o", default="demo_output")
    demo.add_argument("--n-measured", type=int, default=150)
    demo.add_argument("--n-candidates", type=int, default=1000)
    demo.add_argument("--representation", choices=("esm2", "physchem"), default="physchem")
    demo.add_argument("--policy", choices=("top_ranked", "diversity"), default="top_ranked")
    demo.add_argument("--rank-variability", type=int, default=10)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=cmd_demo)

    # benchmark
    benchmark = subparsers.add_parser(
        "benchmark", help="replay a measured dataset and compare strategies"
    )
    benchmark.add_argument("--campaign", required=True)
    benchmark.add_argument("--measurements", required=True)
    benchmark.add_argument("--n-initial", type=int, default=48)
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--rounds", type=int, default=5)
    benchmark.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    benchmark.add_argument("--output", "-o", default=None)
    _add_suite_arguments(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    # app
    app = subparsers.add_parser("app", help="launch the Streamlit interface")
    app.add_argument("--port", type=int, default=None)
    app.set_defaults(func=cmd_app)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        return int(args.func(args) or 0)
    except AssayPilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

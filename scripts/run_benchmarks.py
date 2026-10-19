#!/usr/bin/env python3
"""Run the replay benchmark over every converted assay in a data directory.

    python scripts/fetch_proteingym.py --out data/proteingym --assay SOME_ASSAY
    python scripts/run_benchmarks.py --data data/proteingym/campaigns --out results/

Writes, per assay: ``replay_rounds.csv`` (every round of every strategy and
seed), ``replay_summary.csv`` (final round, averaged over seeds), and
``replay_config.json``. Writes, across assays: ``all_rounds.csv``,
``per_assay_summary.csv``, and ``aggregate_summary.csv``.

Per-assay results are always written alongside the aggregate, so one large
assay cannot be mistaken for a general result. Assays where nothing beat the
baseline stay in the output: removing them would turn the benchmark into a
selection artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from assaypilot import Campaign, load_measurements
from assaypilot.benchmarks.replay import ReplayConfig, compare_strategies, replay_campaign
from assaypilot.models.registry import SuiteConfig
from assaypilot.reporting.manifest import environment_fingerprint

#: Above this share of the pool, the "hit" threshold is not isolating a top
#: slice — usually because many variants tie at a floor or ceiling value — and
#: hit counts stop discriminating between strategies.
MAX_INFORMATIVE_BASE_RATE = 0.25


def discover(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find (name, campaign.json, measurements.csv) triples under ``data_dir``."""
    found = []
    for campaign_path in sorted(data_dir.glob("*/campaign.json")):
        measurements = campaign_path.parent / "measurements.csv"
        if measurements.exists():
            found.append((campaign_path.parent.name, campaign_path, measurements))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory of converted assays")
    parser.add_argument("--out", required=True, help="directory to write results into")
    parser.add_argument("--n-initial", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--representation", choices=("esm2", "physchem"), default="esm2")
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--esm-revision",
        default=None,
        help="pin the checkpoint revision; results are not reproducible without it",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.9,
        help="hit threshold, as a quantile of the assay's measured values",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assays = discover(data_dir)
    if not assays:
        print(f"no assays found under {data_dir} (expected <name>/campaign.json)", file=sys.stderr)
        return 1

    if args.representation == "esm2" and not args.esm_revision:
        print(
            "warning: --esm-revision not given; manifests will record 'unpinned' and the"
            " results will not be exactly reproducible.",
            file=sys.stderr,
        )

    suite = SuiteConfig(
        representation=args.representation,
        esm_checkpoint=args.esm_checkpoint,
        esm_revision=args.esm_revision,
        cache_dir=args.cache_dir,
    )
    config = ReplayConfig(
        n_initial=args.n_initial,
        batch_size=args.batch_size,
        n_rounds=args.rounds,
        seeds=tuple(args.seeds),
        suite=suite,
    )

    all_rounds: list[pd.DataFrame] = []
    per_assay: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for name, campaign_path, measurements_path in assays:
        print(f"\n=== {name} ===")
        try:
            campaign = Campaign.from_json(campaign_path)
            measurements = load_measurements(measurements_path, campaign, strict=False)
            if campaign.success_threshold is None:
                threshold = float(
                    measurements.frame["value"].quantile(
                        args.threshold_quantile
                        if campaign.objective == "maximize"
                        else 1 - args.threshold_quantile
                    )
                )
                campaign = Campaign.from_dict(
                    {**campaign.to_dict(), "success_threshold": threshold}
                )
                print(
                    f"  hit threshold set to the {args.threshold_quantile:.0%} quantile"
                    f" ({threshold:.4g}) for this replay"
                )

            # What share of the pool counts as a hit? A strategy's hit count
            # means nothing without it: where ties at a floor or ceiling value
            # push the "top decile" up to most of the pool, random selection
            # scores highly and the metric cannot separate anything.
            pool_values = measurements.frame.groupby("sequence")["value"].mean()
            hit_base_rate = float(pool_values.map(campaign.is_hit).mean())
            informative = hit_base_rate <= MAX_INFORMATIVE_BASE_RATE
            print(
                f"  {hit_base_rate:.0%} of the pool counts as a hit"
                + (
                    ""
                    if informative
                    else "  <-- threshold does not isolate a top slice;"
                    " hit counts for this assay are uninformative"
                )
            )

            result = replay_campaign(campaign, measurements.frame, config)
            assay_dir = out_dir / name
            assay_dir.mkdir(parents=True, exist_ok=True)
            result.frame.to_csv(assay_dir / "replay_rounds.csv", index=False)
            summary = result.summary()
            summary.insert(0, "assay", name)
            (assay_dir / "replay_config.json").write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            comparison = compare_strategies(result)
            comparison.insert(0, "assay", name)
            comparison.to_csv(assay_dir / "difference_from_random.csv", index=False)

            frame = result.frame.copy()
            frame.insert(0, "assay", name)
            # Assay values live on whatever scale the experiment used — an
            # esterase activity near 50, a fitness near 1 — so they cannot be
            # averaged across assays. The percentile of a value within its own
            # assay's pool can be: "the best variant this strategy chose sits
            # at the 97th percentile of what was available".
            pool = measurements.frame["value"].to_numpy(dtype=float)
            for column in ("best_selected", "best_found"):
                frame[f"{column}_pct"] = [
                    float((pool <= value).mean()) if pd.notna(value) else float("nan")
                    for value in frame[column]
                ]
            final = frame.sort_values("round").groupby(["strategy", "seed"]).tail(1)
            percentiles = (
                final.groupby("strategy")
                .agg(
                    best_selected_pct_mean=("best_selected_pct", "mean"),
                    best_found_pct_mean=("best_found_pct", "mean"),
                )
                .reset_index()
            )
            summary = summary.merge(percentiles, on="strategy", how="left")
            summary["pool_hit_base_rate"] = hit_base_rate
            summary["threshold_informative"] = informative
            summary.to_csv(assay_dir / "replay_summary.csv", index=False)

            all_rounds.append(frame)
            per_assay.append(summary)
            print(summary.to_string(index=False))
        except Exception as exc:  # a failed assay is reported, not hidden
            print(f"  FAILED: {exc}", file=sys.stderr)
            traceback.print_exc(limit=2)
            failures.append({"assay": name, "error": str(exc)})

    if all_rounds:
        pd.concat(all_rounds).to_csv(out_dir / "all_rounds.csv", index=False)
        per_assay_frame = pd.concat(per_assay)
        per_assay_frame.to_csv(out_dir / "per_assay_summary.csv", index=False)

        # The aggregate averages per-assay means, so a large assay does not
        # dominate by contributing more rows. It aggregates only quantities
        # that mean the same thing in every assay: hit counts, which are
        # counts of wells, and percentiles within each assay's own pool. Raw
        # assay values are deliberately absent — averaging an esterase
        # activity of 50 with a fitness of 0.5 produces a number with no
        # interpretation, and whichever assay has the largest units decides it.
        aggregate = (
            per_assay_frame.groupby("strategy")
            .agg(
                assays=("assay", "nunique"),
                best_selected_pct_mean=("best_selected_pct_mean", "mean"),
                n_hits_mean_of_assay_means=("n_hits_mean", "mean"),
                seconds_mean=("seconds_mean", "mean"),
            )
            .reset_index()
        )
        # Assays whose threshold failed to isolate a top slice are kept in the
        # per-assay table and dropped from this column, with the count stated,
        # rather than quietly averaged in or silently removed everywhere.
        usable = per_assay_frame[per_assay_frame["threshold_informative"]]
        informative = (
            usable.groupby("strategy")
            .agg(
                assays_with_usable_threshold=("assay", "nunique"),
                n_hits_usable_assays_only=("n_hits_mean", "mean"),
            )
            .reset_index()
        )
        aggregate = aggregate.merge(informative, on="strategy", how="left").sort_values(
            "n_hits_usable_assays_only", ascending=False
        )
        aggregate.to_csv(out_dir / "aggregate_summary.csv", index=False)
        print("\n=== aggregate (mean of per-assay means; comparable quantities only) ===")
        print(aggregate.to_string(index=False))

    (out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": config.to_dict(),
                "assays": [name for name, _, _ in assays],
                "failures": failures,
                "environment": environment_fingerprint(),
                "caveats": [
                    "Retrospective replay: selection is restricted to variants whose outcome"
                    " the dataset already contains.",
                    "Published assay sequences may overlap protein-language-model pretraining"
                    " corpora; these are not unseen-sequence evaluations.",
                    "Per-assay results are reported alongside the aggregate; the aggregate is"
                    " a mean of per-assay means.",
                    "The aggregate covers only quantities that are comparable across assays"
                    " — hit counts and within-assay percentiles. Raw assay values are not"
                    " averaged across assays, because their units differ.",
                    "Assays where the hit threshold covers more than"
                    f" {MAX_INFORMATIVE_BASE_RATE:.0%} of the pool are reported per assay but"
                    " excluded from the aggregate hit column, because random selection scores"
                    " highly there and the metric separates nothing.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote results to {out_dir}")
    return 0 if all_rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())

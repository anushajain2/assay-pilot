#!/usr/bin/env python3
"""Measure how long AssayPilot takes on this machine, and write the numbers down.

Runtime depends on sequence length, candidate pool size, hardware, and
checkpoint size, so AssayPilot promises nothing about speed. This script
measures it instead:

    python scripts/profile_runtime.py --out results/profile
    python scripts/profile_runtime.py --representation esm2 --esm-revision <commit>

It writes ``profile.csv`` and ``profile.md`` — a table you can paste into a
report, with the hardware and package versions recorded next to it.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from assaypilot import load_candidates, load_measurements, plan_batch
from assaypilot.datasets import simulate_campaign
from assaypilot.features.base import EmbeddingCache, embed_sequences
from assaypilot.models.registry import SuiteConfig, build_backend


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame as a Markdown table without pulling in another dependency."""
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def peak_memory_mb() -> float:
    """Peak resident set size, in MB (ru_maxrss is bytes on macOS, KB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/profile")
    parser.add_argument("--representation", choices=("esm2", "physchem"), default="physchem")
    parser.add_argument("--esm-checkpoint", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--esm-revision", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 5000],
        help="candidate pool sizes to profile",
    )
    parser.add_argument("--n-measured", type=int, default=150)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "embedding_cache"

    suite = SuiteConfig(
        representation=args.representation,
        esm_checkpoint=args.esm_checkpoint,
        esm_revision=args.esm_revision,
        device=args.device,
        cache_dir=str(cache_dir),
    )

    rows: list[dict[str, Any]] = []
    largest = max(args.pool_sizes)

    print(f"Generating a simulated campaign: {args.n_measured} measured, {largest} candidates")
    reference_pool = simulate_campaign(
        n_measured=args.n_measured,
        n_candidates=largest,
        n_variable_positions=24,
        seed=11,
    )
    campaign = reference_pool.campaign

    # Cold embedding throughput, measured with an empty cache.
    backend = build_backend(suite)
    sequences = reference_pool.candidates["sequence"].tolist()[:200]
    cold_cache = EmbeddingCache(cache_dir / "cold", enabled=False)
    start = time.perf_counter()
    embed_sequences(backend, sequences, cache=cold_cache)
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "step": "embedding (cold, no cache)",
            "n_sequences": len(sequences),
            "seconds": round(elapsed, 3),
            "sequences_per_second": round(len(sequences) / elapsed, 1) if elapsed else None,
            "pool_size": None,
        }
    )
    print(f"  embedding: {len(sequences) / elapsed:.1f} sequences/s")

    measurements = load_measurements(reference_pool.measurements, campaign, strict=False)

    for pool_size in sorted(args.pool_sizes):
        subset = reference_pool.candidates.head(pool_size)
        candidates = load_candidates(subset, campaign, strict=False)

        for cache_state in ("cold", "warm"):
            suite_run = SuiteConfig(
                representation=suite.representation,
                esm_checkpoint=suite.esm_checkpoint,
                esm_revision=suite.esm_revision,
                device=suite.device,
                cache_dir=str(cache_dir),
                use_cache=cache_state == "warm",
            )
            start = time.perf_counter()
            plan = plan_batch(
                campaign,
                candidates,
                measurements,
                suite=suite_run,
                n_bootstrap=200,
                seed=0,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "step": f"plan ({cache_state} cache)",
                    "pool_size": pool_size,
                    "n_measured": len(measurements),
                    "batch_size": args.batch_size,
                    "seconds": round(elapsed, 3),
                    "method_used": plan.method_key,
                    "peak_memory_mb": round(peak_memory_mb(), 1),
                }
            )
            print(f"  pool {pool_size:>5} ({cache_state} cache): {elapsed:6.2f}s")

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "profile.csv", index=False)

    environment = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "representation": args.representation,
        "checkpoint": args.esm_checkpoint if args.representation == "esm2" else None,
        "revision": args.esm_revision or ("unpinned" if args.representation == "esm2" else None),
        "sequence_length": campaign.length,
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    lines = [
        "# Measured runtime",
        "",
        f"Measured {environment['measured_at']} on {environment['platform']}"
        f" ({environment['processor']}), Python {environment['python']}.",
        f"Representation: `{args.representation}`"
        + (
            f", checkpoint `{args.esm_checkpoint}` at revision `{environment['revision']}`."
            if args.representation == "esm2"
            else "."
        ),
        f"Reference length {campaign.length} residues, {len(measurements)} measurements.",
        "",
        "These are measurements from one machine, not a performance claim. Re-run",
        "`scripts/profile_runtime.py` to get the numbers for yours.",
        "",
        _markdown_table(frame),
        "",
    ]
    (out / "profile.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out / 'profile.csv'} and {out / 'profile.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

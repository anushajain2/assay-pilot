#!/usr/bin/env python3
"""Regenerate the bundled example campaign in ``examples/simulated-enzyme``.

The example is **synthetic**: it is produced by
``assaypilot.datasets.simulate_campaign`` with a fixed seed and describes no
real protein. It exists so that a reader can run the whole workflow in one
command without downloading anything.

    python scripts/make_example.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from assaypilot.datasets import simulate_campaign
from assaypilot.sequences import mutation_string

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "examples" / "simulated-enzyme"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--n-measured", type=int, default=150)
    parser.add_argument("--n-candidates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    out = Path(args.out)
    simulated = simulate_campaign(
        n_measured=args.n_measured,
        n_candidates=args.n_candidates,
        n_variable_positions=18,
        seed=args.seed,
    )

    out.mkdir(parents=True, exist_ok=True)
    simulated.campaign.to_json(out / "campaign.json")

    # Measurements carry full sequences; candidates use mutation notation. The
    # two files together show both accepted input styles.
    simulated.measurements.to_csv(out / "measurements.csv", index=False)

    candidates = simulated.candidates.copy()
    reference = simulated.campaign.reference_sequence
    candidates["mutations"] = [mutation_string(reference, s) for s in candidates["sequence"]]
    candidates[["candidate_id", "mutations"]].to_csv(out / "candidates.csv", index=False)

    (out / "reference.fasta").write_text(
        f">{simulated.campaign.campaign_id} simulated reference (not a real protein)\n"
        + "\n".join(reference[i : i + 60] for i in range(0, len(reference), 60))
        + "\n",
        encoding="utf-8",
    )

    simulated.truth.to_csv(out / "ground_truth.csv", index=False)
    (out / "README.md").write_text(
        simulated.description + "\n## Run it\n\n"
        "```bash\n"
        "assaypilot validate \\\n"
        "  --campaign examples/simulated-enzyme/campaign.json \\\n"
        "  --measurements examples/simulated-enzyme/measurements.csv \\\n"
        "  --candidates examples/simulated-enzyme/candidates.csv\n\n"
        "assaypilot plan \\\n"
        "  --campaign examples/simulated-enzyme/campaign.json \\\n"
        "  --measurements examples/simulated-enzyme/measurements.csv \\\n"
        "  --candidates examples/simulated-enzyme/candidates.csv \\\n"
        "  --representation physchem \\\n"
        "  --rank-variability 20 \\\n"
        "  -o example_plan\n"
        "```\n\n"
        "`candidates.csv` uses mutation notation and `measurements.csv` uses full\n"
        "sequences, so the example exercises both input styles.\n\n"
        "`ground_truth.csv` holds the noise-free value behind every sequence. The\n"
        "planner never reads it; it is there so you can check for yourself how the\n"
        "selected batch compares with what was achievable.\n\n"
        "Regenerate with `python scripts/make_example.py`.\n",
        encoding="utf-8",
    )

    print(f"Wrote the example campaign to {out}")
    for path in sorted(out.iterdir()):
        print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

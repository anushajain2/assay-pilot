#!/usr/bin/env python3
"""Fetch ProteinGym substitution assays and convert them to AssayPilot inputs.

AssayPilot does not redistribute public datasets. This script downloads them
into a directory you choose and records which release was used, so a benchmark
run can say exactly what it was run on.

    python scripts/fetch_proteingym.py --out data/proteingym --list
    python scripts/fetch_proteingym.py --out data/proteingym --assay BLAT_ECOLX_Stiffler_2015

Downloads are large. By default only the assays you name are converted.

Note on interpretation: published assay sequences may overlap the pretraining
corpora of protein language models, so results on them must not be described as
evaluation on sequences a model has never seen.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

#: ProteinGym publishes bundles of DMS assays; the URL is versioned by release.
DEFAULT_BUNDLE_URL = (
    "https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip"
)


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"already downloaded: {destination}")
        return destination
    print(f"downloading {url}\n        -> {destination}")

    def progress(block: int, block_size: int, total: int) -> None:
        if total > 0 and block % 200 == 0:
            done = min(block * block_size, total)
            print(f"  {done / 1e6:7.1f} MB / {total / 1e6:.1f} MB", end="\r")

    urlretrieve(url, destination, reporthook=progress)  # noqa: S310 - user-supplied URL
    print()
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="directory to download and convert into")
    parser.add_argument("--url", default=DEFAULT_BUNDLE_URL, help="dataset bundle URL")
    parser.add_argument("--assay", action="append", default=None, help="assay file stem to convert")
    parser.add_argument(
        "--list", action="store_true", help="list the assays in the bundle and exit"
    )
    parser.add_argument(
        "--objective",
        choices=("maximize", "minimize"),
        default="maximize",
        help="direction of improvement for the converted campaigns",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    archive = download(args.url, out / "DMS_ProteinGym_substitutions.zip")

    with zipfile.ZipFile(archive) as bundle:
        members = [n for n in bundle.namelist() if n.endswith(".csv")]
        if args.list:
            for name in sorted(members):
                print(Path(name).stem)
            print(f"\n{len(members)} assay file(s) in {archive.name}")
            return 0

        if not args.assay:
            parser.error("name at least one --assay, or pass --list to see what is available")

        wanted = {a.lower() for a in args.assay}
        selected = [n for n in members if Path(n).stem.lower() in wanted]
        missing = wanted - {Path(n).stem.lower() for n in selected}
        if missing:
            print(f"not found in the bundle: {', '.join(sorted(missing))}", file=sys.stderr)
        if not selected:
            return 1

        extracted = out / "raw"
        extracted.mkdir(parents=True, exist_ok=True)
        for name in selected:
            bundle.extract(name, extracted)

    from assaypilot.datasets import load_proteingym_assay

    converted = out / "campaigns"
    converted.mkdir(parents=True, exist_ok=True)
    record: list[dict[str, str]] = []

    for name in selected:
        source = extracted / name
        stem = Path(name).stem
        campaign, measurements = load_proteingym_assay(source, objective=args.objective)
        directory = converted / stem
        directory.mkdir(parents=True, exist_ok=True)
        campaign.to_json(directory / "campaign.json")
        measurements.to_csv(directory / "measurements.csv", index=False)
        record.append(
            {
                "assay": stem,
                "source_file": str(source),
                "campaign": str(directory / "campaign.json"),
                "measurements": str(directory / "measurements.csv"),
                "n_variants": str(len(measurements)),
                "reference_length": str(campaign.length),
            }
        )
        print(f"converted {stem}: {len(measurements)} variants, reference {campaign.length} aa")

    provenance = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": args.url,
        "archive": str(archive),
        "assays": record,
        "note": (
            "Public measured data, not redistributed by AssayPilot. Published assay"
            " sequences may overlap protein-language-model pretraining corpora; do not"
            " describe results on them as evaluation on unseen sequences."
        ),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The run manifest: everything needed to reproduce or audit a plan.

A manifest is written for every plan AssayPilot produces. It records the
inputs by checksum, the transformations applied, the partition assignments,
the random seeds, the model settings including the pinned checkpoint
revision, the package versions, the predictions, every exclusion, and the
selected variants.

Two rules:

* A model *name* is not a reproducible setting. Whatever identifies a
  checkpoint exactly — a revision hash where one is available — goes in the
  manifest, and an unpinned checkpoint is recorded as ``"unpinned"`` rather
  than being left to look reproducible.
* The manifest distinguishes observed values, predicted values, and selection
  reasons. A number never changes category on its way into a file.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

#: Bumped when the manifest layout changes in a way readers must notice.
MANIFEST_SCHEMA_VERSION = "1.0"

_TRACKED_PACKAGES = (
    "assaypilot",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
)


def file_checksum(path: str | Path) -> dict[str, Any]:
    """SHA-256, size, and name of an input file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "name": path.name,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def frame_checksum(frame, *, name: str) -> dict[str, Any]:
    """Checksum of an in-memory table, for inputs that never touched disk."""
    payload = frame.to_csv(index=False).encode("utf-8")
    return {
        "path": f"<in-memory:{name}>",
        "name": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def environment_fingerprint() -> dict[str, Any]:
    """Python, platform, and the versions of the packages that affect results."""
    versions: dict[str, str] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def assaypilot_version() -> str:
    try:
        return metadata.version("assaypilot")
    except metadata.PackageNotFoundError:  # running from a source checkout
        return "0.1.0+source"


def new_run_id(prefix: str = "run") -> str:
    """A time-ordered, unique run identifier such as ``run-20261018-5f3a1c``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


@dataclass
class RunManifest:
    """Machine-readable record of one planning run."""

    run_id: str
    campaign: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = MANIFEST_SCHEMA_VERSION
    assaypilot_version: str = field(default_factory=assaypilot_version)
    environment: dict[str, Any] = field(default_factory=environment_fingerprint)
    inputs: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    selected: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "assaypilot_version": self.assaypilot_version,
            "environment": self.environment,
            "campaign": self.campaign,
            "inputs": self.inputs,
            "config": self.config,
            "runtime": self.runtime,
            "data": self.data,
            "evaluation": self.evaluation,
            "coverage": self.coverage,
            "selection": self.selection,
            "predictions": self.predictions,
            "exclusions": self.exclusions,
            "selected": self.selected,
            "warnings": self.warnings,
            "notes": self.notes,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def fingerprint(self) -> str:
        """Hash of the reproducibility-relevant fields.

        Two runs with the same fingerprint used the same inputs, configuration,
        seeds, and package versions. Timestamps, the run id, and runtime
        observations such as cache hit counts are excluded, so a deterministic
        rerun matches its predecessor.
        """
        payload = {
            "campaign": self.campaign,
            "inputs": self.inputs,
            "config": self.config,
            "environment": self.environment,
        }
        blob = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _json_default(value: Any) -> Any:
    """Make numpy scalars and sets JSON-serialisable."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)

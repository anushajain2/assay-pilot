"""Readers for the two inputs a campaign needs: measurements and candidates.

Both readers produce a canonical table plus a :class:`ValidationReport`. The
canonical columns are fixed so that every downstream component — splits,
models, selection, exports — can rely on them:

``variant_id`` / ``candidate_id``, ``sequence``, ``seq_hash``, ``mutations``,
``n_substitutions`` and, for measurements, ``value``, ``value_raw``,
``censoring``, ``condition_id``, ``round_id``, ``replicate_id``, ``plate_id``.

``seq_hash`` is what keeps replicates and identical sequences in the same
evaluation partition, which is the leakage rule version one cares about most.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..campaign import Campaign
from ..errors import InputError, SequenceError
from ..sequences import (
    parse_mutation_string,
    sequence_hash,
    substitutions,
    validate_sequence,
)
from .fasta import read_fasta
from .validation import ExclusionCode, ValidationReport

Transform = Literal["none", "log10", "log2", "ln", "zscore"]

MEASUREMENT_COLUMNS = (
    "variant_id",
    "sequence",
    "seq_hash",
    "mutations",
    "n_substitutions",
    "value",
    "value_raw",
    "censoring",
    "condition_id",
    "round_id",
    "replicate_id",
    "plate_id",
    "source_row",
)

CANDIDATE_COLUMNS = (
    "candidate_id",
    "sequence",
    "seq_hash",
    "mutations",
    "n_substitutions",
    "source_row",
)

_CENSOR_PREFIXES = {"<": "left", "<=": "left", ">": "right", ">=": "right"}


def _first_present(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _read_table(source: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if isinstance(source, pd.DataFrame):
        return source.copy(), "<dataframe>"
    path = Path(source)
    if not path.exists():
        raise InputError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise InputError(
            f"unsupported file type {suffix!r} for {path.name};"
            " use .csv, .tsv, .parquet, or (for candidates) .fasta"
        )
    if frame.empty:
        raise InputError(f"{path.name} has no data rows")
    return frame, path.name


def _parse_censored(raw: Any) -> tuple[float | None, str]:
    """Return ``(value, censoring)`` for a possibly censored cell.

    ``"<0.01"`` becomes ``(0.01, "left")``; ``"12.5"`` becomes ``(12.5, "none")``.
    Censored values are kept in the table but excluded from regression in
    version one rather than being treated as exact.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None, "none"
    text = str(raw).strip()
    if not text or text.lower() in {"na", "nan", "none", "null", ""}:
        return None, "none"
    for prefix in ("<=", ">=", "<", ">"):
        if text.startswith(prefix):
            try:
                return float(text[len(prefix) :].strip()), _CENSOR_PREFIXES[prefix]
            except ValueError:
                return None, "none"
    try:
        return float(text), "none"
    except ValueError:
        return None, "none"


def _resolve_sequence(
    row: pd.Series,
    *,
    campaign: Campaign,
    sequence_col: str | None,
    mutation_col: str | None,
    identifier: str,
) -> str:
    """Build a validated full-length sequence from a sequence or mutation column."""
    if sequence_col is not None:
        raw = row.get(sequence_col)
        if raw is not None and str(raw).strip() and str(raw).strip().lower() != "nan":
            return validate_sequence(
                str(raw), reference=campaign.reference_sequence, label=f"{identifier} sequence"
            )
    if mutation_col is not None:
        raw = row.get(mutation_col)
        if raw is not None and str(raw).strip() and str(raw).strip().lower() != "nan":
            return parse_mutation_string(str(raw), campaign.reference_sequence)
    raise SequenceError(f"{identifier} has neither a sequence nor mutation notation")


@dataclass
class MeasurementSet:
    """Validated measurements for one campaign, with the report that produced them."""

    campaign: Campaign
    frame: pd.DataFrame
    report: ValidationReport
    transform: Transform = "none"
    transform_detail: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    @property
    def sequences(self) -> list[str]:
        return self.frame["sequence"].tolist()

    @property
    def values(self) -> np.ndarray:
        return self.frame["value"].to_numpy(dtype=float)

    @property
    def groups(self) -> np.ndarray:
        """Group labels (sequence hashes) that keep replicates in one partition."""
        return self.frame["seq_hash"].to_numpy()

    @property
    def measured_hashes(self) -> set[str]:
        return set(self.frame["seq_hash"].tolist())

    @property
    def n_unique_sequences(self) -> int:
        return int(self.frame["seq_hash"].nunique())

    def rounds(self) -> list[Any]:
        """Sorted round labels present, excluding missing ones."""
        if "round_id" not in self.frame:
            return []
        values = self.frame["round_id"].dropna().unique().tolist()
        return sorted(values, key=_round_sort_key)

    def regression_frame(self) -> pd.DataFrame:
        """Rows usable for fitting: finite, uncensored values only."""
        frame = self.frame
        usable = frame["censoring"].eq("none") & np.isfinite(frame["value"].to_numpy(dtype=float))
        return frame.loc[usable].reset_index(drop=True)

    def for_regression(self) -> MeasurementSet:
        """A copy restricted to rows usable for fitting (see :meth:`regression_frame`)."""
        regression = self.regression_frame()
        dropped = len(self.frame) - len(regression)
        report = ValidationReport(
            source=self.report.source,
            n_rows_read=self.report.n_rows_read,
            n_rows_kept=len(regression),
            exclusions=list(self.report.exclusions),
            notes=list(self.report.notes),
            stats=dict(self.report.stats),
        )
        if dropped:
            report.note(
                f"{dropped} censored or non-finite measurement(s) kept in the data but excluded"
                " from model fitting in version one"
            )
        return replace(self, frame=regression, report=report)

    def replicate_summary(self) -> pd.DataFrame:
        """Per-sequence replicate counts and spread, for the data review screen."""
        grouped = self.frame.groupby("seq_hash").agg(
            variant_id=("variant_id", "first"),
            mutations=("mutations", "first"),
            n_measurements=("value", "size"),
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        grouped["range_value"] = grouped["max_value"] - grouped["min_value"]
        return grouped.sort_values("n_measurements", ascending=False).reset_index()

    def conflicting_replicates(self, *, relative_spread: float = 0.5) -> pd.DataFrame:
        """Sequences whose replicates disagree by more than ``relative_spread``.

        The disagreement is exposed, never resolved: the raw values stay in the
        table and no replicate is dropped.
        """
        summary = self.replicate_summary()
        multi = summary[summary["n_measurements"] > 1].copy()
        if multi.empty:
            return multi
        scale = multi["mean_value"].abs().replace(0, np.nan)
        multi["relative_spread"] = (multi["range_value"] / scale).fillna(np.inf)
        return multi[multi["relative_spread"] > relative_spread].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_measurements": len(self.frame),
            "n_unique_sequences": self.n_unique_sequences,
            "rounds": [str(r) for r in self.rounds()],
            "transform": self.transform,
            "transform_detail": dict(self.transform_detail),
            "value_summary": {
                "min": float(np.nanmin(self.values)) if len(self.frame) else None,
                "max": float(np.nanmax(self.values)) if len(self.frame) else None,
                "mean": float(np.nanmean(self.values)) if len(self.frame) else None,
            },
        }


@dataclass
class CandidateSet:
    """A validated pool of untested candidates for one campaign."""

    campaign: Campaign
    frame: pd.DataFrame
    report: ValidationReport

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def sequences(self) -> list[str]:
        return self.frame["sequence"].tolist()

    @property
    def ids(self) -> list[str]:
        return self.frame["candidate_id"].tolist()

    def to_dict(self) -> dict[str, Any]:
        n_subs = self.frame["n_substitutions"]
        return {
            "n_candidates": len(self.frame),
            "substitutions": {
                "min": int(n_subs.min()) if len(self.frame) else None,
                "max": int(n_subs.max()) if len(self.frame) else None,
                "median": float(n_subs.median()) if len(self.frame) else None,
            },
        }


def _round_sort_key(value: Any) -> tuple[int, float, str]:
    """Sort rounds numerically when possible, alphabetically otherwise."""
    text = str(value)
    try:
        return (0, float(text), "")
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            return (0, float(digits), text)
        return (1, 0.0, text)


def load_measurements(
    source: str | Path | pd.DataFrame,
    campaign: Campaign,
    *,
    strict: bool = True,
    sequence_column: str | None = None,
    mutation_column: str | None = None,
    value_column: str | None = None,
    id_column: str | None = None,
    condition_column: str | None = None,
    round_column: str | None = None,
    include_other_conditions: bool = False,
) -> MeasurementSet:
    """Read and validate a measurements table.

    The file needs an identifier, a sequence (or mutation notation checked
    against the reference), a numeric value, and a condition label. Rounds,
    replicates, and plates are optional but unlock round-based evaluation and
    plate-aware review.

    Parameters
    ----------
    strict:
        When ``True`` (default) an input with blocking problems — invalid
        characters, length mismatches, unparseable mutation notation, duplicate
        identifiers — raises :class:`~assaypilot.errors.InputError` listing
        them. When ``False`` those rows are excluded with a recorded reason and
        the rest are returned, which is what the review screen uses.
    include_other_conditions:
        When ``False`` (default) rows whose condition differs from the campaign
        condition are excluded rather than pooled. Different conditions belong
        in a different campaign.
    """
    frame, source_name = _read_table(source)
    report = ValidationReport(source=source_name, n_rows_read=len(frame))

    id_col = id_column or _first_present(frame, ["variant_id", "id", "name", "variant"])
    seq_col = sequence_column or _first_present(
        frame, ["sequence", "seq", "aa_sequence", "protein_sequence"]
    )
    mut_col = mutation_column or _first_present(
        frame, ["mutations", "mutation", "variant_notation", "muts"]
    )
    val_col = value_column or _first_present(
        frame, ["value", "measurement", "fitness", "activity", "score"]
    )
    cond_col = condition_column or _first_present(frame, ["condition_id", "condition"])
    round_col = round_column or _first_present(frame, ["round_id", "round"])
    rep_col = _first_present(frame, ["replicate_id", "replicate", "rep"])
    plate_col = _first_present(frame, ["plate_id", "plate"])

    if seq_col is None and mut_col is None:
        raise InputError(
            f"{source_name} needs a 'sequence' column or a 'mutations' column"
            " (mutation notation is checked against the campaign reference)"
        )
    if val_col is None:
        raise InputError(f"{source_name} needs a 'value' column with the measured outcome")

    rows: list[dict[str, Any]] = []
    id_to_hash: dict[str, str] = {}
    n_censored = 0

    for position, (_, row) in enumerate(frame.iterrows(), start=2):  # start=2: header is row 1
        raw_id = row.get(id_col) if id_col else None
        identifier = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else ""
        if not identifier:
            if id_col is None:
                identifier = f"row{position}"
            else:
                report.add(
                    ExclusionCode.MISSING_ID, f"row{position}", "variant_id is empty", row=position
                )
                continue

        condition = str(row.get(cond_col)).strip() if cond_col else campaign.condition_id
        if condition in ("", "nan", "None"):
            condition = campaign.condition_id
        if condition != campaign.condition_id and not include_other_conditions:
            report.add(
                ExclusionCode.OTHER_CONDITION,
                identifier,
                f"condition {condition!r} differs from campaign condition"
                f" {campaign.condition_id!r}; create a separate campaign for it rather than"
                " pooling the measurements",
                row=position,
            )
            continue

        try:
            sequence = _resolve_sequence(
                row,
                campaign=campaign,
                sequence_col=seq_col,
                mutation_col=mut_col,
                identifier=identifier,
            )
        except SequenceError as exc:
            message = str(exc)
            if "length" in message:
                code = ExclusionCode.LENGTH_MISMATCH
            elif "neither a sequence nor" in message:
                code = ExclusionCode.MISSING_SEQUENCE
            elif "contains" in message:
                code = ExclusionCode.INVALID_CHARACTERS
            else:
                code = ExclusionCode.MUTATION_PARSE_ERROR
            report.add(code, identifier, message, row=position)
            continue

        seq_hash = sequence_hash(sequence)
        previous = id_to_hash.get(identifier)
        if previous is not None and previous != seq_hash:
            report.add(
                ExclusionCode.DUPLICATE_ID,
                identifier,
                "the same variant_id appears with two different sequences; give each distinct"
                " sequence its own identifier",
                row=position,
            )
            continue
        id_to_hash[identifier] = seq_hash

        value, censoring = _parse_censored(row.get(val_col))
        if value is None:
            raw_text = str(row.get(val_col)).strip()
            if raw_text in ("", "nan", "None", "NA", "na"):
                report.add(ExclusionCode.MISSING_VALUE, identifier, "value is empty", row=position)
            else:
                report.add(
                    ExclusionCode.NON_NUMERIC_VALUE,
                    identifier,
                    f"value {raw_text!r} is not a number and is not censoring notation"
                    " such as '<0.01'",
                    row=position,
                )
            continue
        if censoring != "none":
            n_censored += 1

        subs = substitutions(campaign.reference_sequence, sequence)
        rows.append(
            {
                "variant_id": identifier,
                "sequence": sequence,
                "seq_hash": seq_hash,
                "mutations": ":".join(str(s) for s in subs) if subs else "WT",
                "n_substitutions": len(subs),
                "value": float(value),
                "value_raw": float(value),
                "censoring": censoring,
                "condition_id": condition,
                "round_id": _clean_optional(row.get(round_col)) if round_col else None,
                "replicate_id": _clean_optional(row.get(rep_col)) if rep_col else None,
                "plate_id": _clean_optional(row.get(plate_col)) if plate_col else None,
                "source_row": position,
            }
        )

    table = pd.DataFrame(rows, columns=list(MEASUREMENT_COLUMNS))
    report.n_rows_kept = len(table)

    if not table.empty:
        duplicated_sequences = int(table["seq_hash"].duplicated().sum())
        if duplicated_sequences:
            report.note(
                f"{duplicated_sequences} row(s) repeat a sequence already present; they are kept"
                " as replicates and always stay in the same evaluation partition"
            )
        if n_censored:
            report.note(
                f"{n_censored} censored value(s) kept in the table and excluded from model"
                " fitting in version one"
            )
        report.stats = {
            "n_unique_sequences": int(table["seq_hash"].nunique()),
            "n_rounds": int(table["round_id"].nunique(dropna=True)),
            "conditions": sorted({str(c) for c in table["condition_id"]}),
            "max_substitutions": int(table["n_substitutions"].max()),
        }

    if strict:
        report.raise_if_blocking()
    return MeasurementSet(campaign=campaign, frame=table, report=report)


def _clean_optional(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "None", "NaN"):
        return None
    return text


def load_candidates(
    source: str | Path | pd.DataFrame,
    campaign: Campaign,
    *,
    strict: bool = True,
    sequence_column: str | None = None,
    mutation_column: str | None = None,
    id_column: str | None = None,
) -> CandidateSet:
    """Read and validate a candidate pool from CSV/TSV/Parquet or FASTA.

    Duplicate sequences inside the pool are excluded (the first occurrence is
    kept) so that a batch never spends two wells on the same construct.
    """
    source_is_fasta = not isinstance(source, pd.DataFrame) and Path(source).suffix.lower() in {
        ".fa",
        ".fasta",
        ".faa",
        ".fna",
    }

    if source_is_fasta:
        records = read_fasta(source)
        frame = pd.DataFrame(
            {
                "candidate_id": [r.identifier for r in records],
                "sequence": [r.sequence for r in records],
            }
        )
        source_name = Path(source).name
    else:
        frame, source_name = _read_table(source)

    report = ValidationReport(source=source_name, n_rows_read=len(frame))

    id_col = id_column or _first_present(frame, ["candidate_id", "id", "name", "variant_id"])
    seq_col = sequence_column or _first_present(frame, ["sequence", "seq", "aa_sequence"])
    mut_col = mutation_column or _first_present(
        frame, ["mutations", "mutation", "variant_notation"]
    )

    if seq_col is None and mut_col is None:
        raise InputError(
            f"{source_name} needs a 'sequence' column or a 'mutations' column for candidates"
        )

    rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()

    for position, (_, row) in enumerate(frame.iterrows(), start=2):
        raw_id = row.get(id_col) if id_col else None
        identifier = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else ""
        if not identifier:
            identifier = f"cand{position - 1:05d}"
        if identifier in seen_ids:
            report.add(
                ExclusionCode.DUPLICATE_ID,
                identifier,
                "candidate_id is used more than once; identifiers must be unique so that"
                " results can be matched back after the experiment",
                row=position,
            )
            continue

        try:
            sequence = _resolve_sequence(
                row,
                campaign=campaign,
                sequence_col=seq_col,
                mutation_col=mut_col,
                identifier=f"candidate {identifier}",
            )
        except SequenceError as exc:
            message = str(exc)
            if "length" in message:
                code = ExclusionCode.LENGTH_MISMATCH
            elif "neither a sequence nor" in message:
                code = ExclusionCode.MISSING_SEQUENCE
            elif "contains" in message:
                code = ExclusionCode.INVALID_CHARACTERS
            else:
                code = ExclusionCode.MUTATION_PARSE_ERROR
            report.add(code, identifier, message, row=position)
            continue

        seq_hash = sequence_hash(sequence)
        if seq_hash in seen_hashes:
            report.add(
                ExclusionCode.DUPLICATE_SEQUENCE,
                identifier,
                f"identical to candidate {seen_hashes[seq_hash]} already in the pool",
                row=position,
            )
            continue
        seen_hashes[seq_hash] = identifier
        seen_ids.add(identifier)

        subs = substitutions(campaign.reference_sequence, sequence)
        rows.append(
            {
                "candidate_id": identifier,
                "sequence": sequence,
                "seq_hash": seq_hash,
                "mutations": ":".join(str(s) for s in subs) if subs else "WT",
                "n_substitutions": len(subs),
                "source_row": position,
            }
        )

    table = pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))
    report.n_rows_kept = len(table)
    if not table.empty:
        report.stats = {
            "max_substitutions": int(table["n_substitutions"].max()),
            "median_substitutions": float(table["n_substitutions"].median()),
        }
    if strict:
        report.raise_if_blocking()
    return CandidateSet(campaign=campaign, frame=table, report=report)


def candidates_from_sequences(
    sequences: Iterable[str],
    campaign: Campaign,
    *,
    prefix: str = "cand",
    strict: bool = True,
) -> CandidateSet:
    """Build a candidate pool from an iterable of sequences (used by tests and demos)."""
    seqs = list(sequences)
    frame = pd.DataFrame(
        {
            "candidate_id": [f"{prefix}{i:05d}" for i in range(1, len(seqs) + 1)],
            "sequence": seqs,
        }
    )
    return load_candidates(frame, campaign, strict=strict)


def apply_transform(measurements: MeasurementSet, kind: Transform) -> MeasurementSet:
    """Apply an approved value transformation, keeping the raw values alongside.

    ``value_raw`` always holds the number as read from the file. Only ``value``
    changes, and the transformation is recorded so it appears in the run
    manifest and the report.
    """
    if kind == "none":
        return measurements

    frame = measurements.frame.copy()
    raw = frame["value_raw"].to_numpy(dtype=float)
    detail: dict[str, Any] = {"kind": kind}

    if kind in {"log10", "log2", "ln"}:
        non_positive = int((raw <= 0).sum())
        if non_positive:
            raise InputError(
                f"{non_positive} measurement(s) are zero or negative, so a {kind} transform is"
                " not defined for them. Offset the data deliberately in your input file, or"
                " keep the raw scale."
            )
        if kind == "log10":
            frame["value"] = np.log10(raw)
        elif kind == "log2":
            frame["value"] = np.log2(raw)
        else:
            frame["value"] = np.log(raw)
    elif kind == "zscore":
        mean = float(np.mean(raw))
        std = float(np.std(raw, ddof=1)) if len(raw) > 1 else 0.0
        if std == 0:
            raise InputError("cannot z-score measurements with zero variance")
        frame["value"] = (raw - mean) / std
        detail.update({"mean": mean, "std": std})
    else:  # pragma: no cover - guarded by the Transform literal
        raise InputError(f"unknown transform {kind!r}")

    report = ValidationReport(
        source=measurements.report.source,
        n_rows_read=measurements.report.n_rows_read,
        n_rows_kept=measurements.report.n_rows_kept,
        exclusions=list(measurements.report.exclusions),
        notes=list(measurements.report.notes),
        stats=dict(measurements.report.stats),
    )
    report.note(f"values transformed with {kind}; raw values retained in 'value_raw'")
    return replace(
        measurements, frame=frame, report=report, transform=kind, transform_detail=detail
    )

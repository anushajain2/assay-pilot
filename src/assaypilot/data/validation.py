"""Exclusion records and validation reports.

Two rules govern this module:

1. Nothing is dropped silently. Every row that does not reach the model is
   recorded as an :class:`Exclusion` with a machine-readable ``code``, a
   human-readable ``detail``, and the identifier of the row it came from.
2. Nothing is repaired silently. Mutation numbering is never shifted to make a
   variant fit the reference, and outliers are never removed on the tool's own
   initiative.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["blocking", "excluded", "warning", "note"]


class ExclusionCode:
    """Stable codes used in reports, manifests, and exports.

    They are strings rather than an enum so that they survive JSON round-trips
    in a run manifest without a custom decoder.
    """

    MISSING_SEQUENCE = "missing_sequence"
    INVALID_CHARACTERS = "invalid_characters"
    LENGTH_MISMATCH = "length_mismatch"
    MUTATION_PARSE_ERROR = "mutation_parse_error"
    MISSING_VALUE = "missing_value"
    NON_NUMERIC_VALUE = "non_numeric_value"
    OTHER_CONDITION = "other_condition"
    MISSING_ID = "missing_id"
    DUPLICATE_ID = "duplicate_id"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    ALREADY_MEASURED = "already_measured"
    LOCKED_POSITION = "locked_position_changed"
    TOO_MANY_SUBSTITUTIONS = "exceeds_max_substitutions"
    CENSORED_VALUE = "censored_value"
    NO_SUBSTITUTIONS = "identical_to_reference"

    #: Codes that mean the input file itself is wrong and should be fixed before
    #: any model is trained. These block an import unless explicitly overridden.
    BLOCKING = frozenset(
        {
            MISSING_SEQUENCE,
            INVALID_CHARACTERS,
            LENGTH_MISMATCH,
            MUTATION_PARSE_ERROR,
            MISSING_ID,
            DUPLICATE_ID,
            NON_NUMERIC_VALUE,
        }
    )

    #: Short explanations shown in reports next to each code.
    DESCRIPTIONS = {
        MISSING_SEQUENCE: "no sequence given and no mutation notation to build one from",
        INVALID_CHARACTERS: "sequence contains characters outside the 20 standard amino acids",
        LENGTH_MISMATCH: "sequence length differs from the campaign reference",
        MUTATION_PARSE_ERROR: "mutation notation could not be checked against the reference",
        MISSING_VALUE: "measurement value is empty",
        NON_NUMERIC_VALUE: "measurement value is not a number",
        OTHER_CONDITION: "row belongs to a different experimental condition",
        MISSING_ID: "identifier is empty",
        DUPLICATE_ID: "identifier is used by rows with different sequences",
        DUPLICATE_SEQUENCE: "sequence already present in the candidate pool",
        ALREADY_MEASURED: "sequence has already been measured in this campaign",
        LOCKED_POSITION: "candidate substitutes a locked residue position",
        TOO_MANY_SUBSTITUTIONS: "more substitutions than the campaign allows",
        CENSORED_VALUE: "value is censored (reported as above or below a limit)",
        NO_SUBSTITUTIONS: "sequence is identical to the reference",
    }


@dataclass(frozen=True)
class Exclusion:
    """One row that did not reach the model, and why."""

    code: str
    identifier: str
    detail: str
    severity: Severity = "excluded"
    row: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "identifier": self.identifier,
            "detail": self.detail,
            "severity": self.severity,
            "row": self.row,
        }


@dataclass
class ValidationReport:
    """What an import contained, what was kept, and what was set aside."""

    source: str = ""
    n_rows_read: int = 0
    n_rows_kept: int = 0
    exclusions: list[Exclusion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        code: str,
        identifier: str,
        detail: str,
        *,
        row: int | None = None,
        severity: Severity | None = None,
    ) -> None:
        """Record one exclusion, inferring its severity from the code if not given."""
        if severity is None:
            severity = "blocking" if code in ExclusionCode.BLOCKING else "excluded"
        self.exclusions.append(
            Exclusion(
                code=code, identifier=str(identifier), detail=detail, severity=severity, row=row
            )
        )

    def note(self, text: str) -> None:
        """Record something the researcher should see that is not an exclusion."""
        self.notes.append(text)

    def extend(self, other: ValidationReport) -> None:
        self.exclusions.extend(other.exclusions)
        self.notes.extend(other.notes)

    @property
    def blocking(self) -> list[Exclusion]:
        return [e for e in self.exclusions if e.severity == "blocking"]

    @property
    def has_blocking(self) -> bool:
        return bool(self.blocking)

    def counts_by_code(self) -> dict[str, int]:
        return dict(Counter(e.code for e in self.exclusions))

    def summary_lines(self) -> list[str]:
        """Human-readable summary, one line per statement, for CLI and reports."""
        lines = [f"{self.n_rows_kept} of {self.n_rows_read} rows usable"]
        if self.source:
            lines[0] += f" from {self.source}"
        for code, count in sorted(self.counts_by_code().items(), key=lambda kv: -kv[1]):
            description = ExclusionCode.DESCRIPTIONS.get(code, code)
            lines.append(f"  {count} x {code}: {description}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return lines

    def raise_if_blocking(self) -> None:
        """Raise :class:`~assaypilot.errors.InputError` if the file must be fixed first."""
        from ..errors import InputError

        blocking = self.blocking
        if not blocking:
            return
        shown = blocking[:10]
        details = "\n".join(f"  row {e.row}: {e.identifier}: {e.detail}" for e in shown)
        more = (
            f"\n  ... and {len(blocking) - len(shown)} more" if len(blocking) > len(shown) else ""
        )
        raise InputError(
            f"{len(blocking)} row(s) in {self.source or 'the input'} cannot be used as given:\n"
            f"{details}{more}\n"
            "Fix the input, or load with strict=False to review the full report and continue"
            " with the remaining rows."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_rows_read": self.n_rows_read,
            "n_rows_kept": self.n_rows_kept,
            "counts_by_code": self.counts_by_code(),
            "exclusions": [e.to_dict() for e in self.exclusions],
            "notes": list(self.notes),
            "stats": dict(self.stats),
        }


def exclusions_frame(exclusions: Iterable[Exclusion]):
    """Return exclusions as a DataFrame, for display and for ``exclusions.csv``."""
    import pandas as pd

    rows = [e.to_dict() for e in exclusions]
    columns = ["row", "identifier", "code", "severity", "detail"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]

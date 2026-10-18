"""Campaign definition: what is being measured, under which conditions, and how
the next batch may be chosen.

A *campaign* is the unit of work in AssayPilot. It pins a reference sequence, a
named assay, its units, the direction of improvement, and the experimental
condition the measurements belong to. Every trained predictor and every
exported plan is tied to one campaign, which is how the tool keeps its claim
narrow: it predicts *this recorded assay outcome under this condition*, not a
general notion of protein fitness.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .errors import CampaignError
from .sequences import normalise_sequence, sequence_hash, validate_sequence

Objective = Literal["maximize", "minimize"]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_id(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not _ID_RE.match(value):
        raise CampaignError(
            f"{field_name} {value!r} must start with a letter or digit and contain only"
            " letters, digits, '_', '.', or '-'"
        )
    return value


@dataclass(frozen=True)
class SelectionSettings:
    """Constraints applied to the candidate pool before and during ranking.

    Attributes
    ----------
    batch_size:
        Number of *candidate* wells to fill. Controls and replicates are
        budgeted separately (see :attr:`control_sequences` and
        :attr:`replicates_per_candidate`) and are reported separately in the
        export, so that a batch of 24 candidates is never confused with 24
        wells.
    locked_positions:
        1-based positions that must keep their reference residue. A candidate
        substituting a locked position is excluded with that reason recorded.
    max_substitutions:
        Maximum number of substitutions from the reference. ``None`` means no
        limit.
    allow_measured_repeats:
        When ``False`` (the default) candidates whose sequence has already been
        measured in this campaign are excluded. Set ``True`` to deliberately
        re-measure.
    diversity_min_distance:
        Hamming-distance threshold used by the ``diversity`` policy. After a
        candidate is selected, candidates closer than this distance to it are
        suppressed. ``None`` disables the constraint.
    control_sequences:
        Identifiers of control wells planned alongside the batch. Recorded for
        budget bookkeeping; they are not selected by the model.
    replicates_per_candidate:
        Planned replicates per selected candidate, used only to report the
        total well count.
    """

    batch_size: int = 24
    locked_positions: tuple[int, ...] = ()
    max_substitutions: int | None = None
    allow_measured_repeats: bool = False
    diversity_min_distance: int | None = None
    control_sequences: tuple[str, ...] = ()
    replicates_per_candidate: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise CampaignError("batch_size must be at least 1")
        if self.replicates_per_candidate < 1:
            raise CampaignError("replicates_per_candidate must be at least 1")
        if self.max_substitutions is not None and self.max_substitutions < 1:
            raise CampaignError("max_substitutions must be at least 1 when set")
        if self.diversity_min_distance is not None and self.diversity_min_distance < 1:
            raise CampaignError("diversity_min_distance must be at least 1 when set")
        for pos in self.locked_positions:
            if pos < 1:
                raise CampaignError(f"locked position {pos} must be 1-based and positive")

    @property
    def total_wells(self) -> int:
        """Candidate wells plus control wells, counting planned replicates."""
        return self.batch_size * self.replicates_per_candidate + len(self.control_sequences)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["locked_positions"] = list(self.locked_positions)
        data["control_sequences"] = list(self.control_sequences)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionSettings:
        data = dict(data)
        data["locked_positions"] = tuple(data.get("locked_positions") or ())
        data["control_sequences"] = tuple(data.get("control_sequences") or ())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class Campaign:
    """A single protein, a single quantitative assay, a single condition.

    Attributes
    ----------
    campaign_id:
        Short identifier, unique within a store.
    reference_sequence:
        The parent sequence. All variants are substitutions of it.
    assay_name:
        What was measured, in the lab's own words, for example
        ``"initial rate, pNPA hydrolysis"``.
    assay_units:
        Units of ``value``, for example ``"s^-1"`` or ``"RFU"``. Recorded so
        that the export never presents a bare number.
    objective:
        ``"maximize"`` or ``"minimize"``. Rank correlation and batch selection
        follow this direction.
    condition_id:
        The experimental context the measurements share, for example
        ``"37C_pH7.4_lysate"``. Measurements from another condition are not
        pooled automatically; they belong to a separate campaign.
    success_threshold:
        Optional value defining a "hit". Must be set before hit rates are
        reported, and is recorded in the manifest so it cannot be chosen after
        seeing results.
    """

    campaign_id: str
    reference_sequence: str
    assay_name: str
    assay_units: str
    objective: Objective = "maximize"
    condition_id: str = "default"
    description: str = ""
    success_threshold: float | None = None
    created_at: str = field(default_factory=_utc_now)
    selection: SelectionSettings = field(default_factory=SelectionSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _check_id(self.campaign_id, "campaign_id"))
        object.__setattr__(self, "condition_id", _check_id(self.condition_id, "condition_id"))
        seq = validate_sequence(self.reference_sequence, label="reference sequence")
        object.__setattr__(self, "reference_sequence", seq)
        if not str(self.assay_name).strip():
            raise CampaignError("assay_name is required: name the measurement that was recorded")
        if not str(self.assay_units).strip():
            raise CampaignError(
                "assay_units is required: state the units of the recorded value, or 'arbitrary'"
            )
        if self.objective not in ("maximize", "minimize"):
            raise CampaignError("objective must be 'maximize' or 'minimize'")
        for pos in self.selection.locked_positions:
            if pos > len(seq):
                raise CampaignError(
                    f"locked position {pos} is outside the reference (1-{len(seq)})"
                )

    @property
    def length(self) -> int:
        """Length of the reference sequence."""
        return len(self.reference_sequence)

    @property
    def reference_hash(self) -> str:
        """Stable hash of the reference sequence, recorded in every manifest."""
        return sequence_hash(self.reference_sequence)

    @property
    def sign(self) -> int:
        """``+1`` when higher values are better, ``-1`` when lower values are better."""
        return 1 if self.objective == "maximize" else -1

    def is_hit(self, value: float) -> bool:
        """Whether a measured value counts as a hit under the declared threshold."""
        if self.success_threshold is None:
            raise CampaignError(
                "no success_threshold is defined for this campaign; define one before"
                " reporting hit rates"
            )
        if self.objective == "maximize":
            return float(value) >= self.success_threshold
        return float(value) <= self.success_threshold

    def with_selection(self, **changes: Any) -> Campaign:
        """Return a copy with updated selection settings."""
        return replace(self, selection=replace(self.selection, **changes))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selection"] = self.selection.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Campaign:
        data = dict(data)
        selection = data.pop("selection", None)
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        if selection is not None:
            kwargs["selection"] = SelectionSettings.from_dict(selection)
        return cls(**kwargs)

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> Campaign:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_fasta(cls, path: str | Path, **kwargs: Any) -> Campaign:
        """Build a campaign whose reference is the first record of a FASTA file."""
        from .data.fasta import read_fasta

        records = read_fasta(path)
        if not records:
            raise CampaignError(f"{path} contains no sequences")
        first = records[0]
        kwargs.setdefault("campaign_id", _slugify(first.identifier))
        kwargs.setdefault("description", first.description)
        return cls(reference_sequence=normalise_sequence(first.sequence), **kwargs)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-")
    return slug or "campaign"

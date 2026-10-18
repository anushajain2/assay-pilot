"""Eligibility filtering: decide what may be ordered, before anything is ranked.

Filtering comes first and is entirely deterministic. A candidate is scored
only after it has passed, which means a high prediction can never pull an
ineligible sequence into a batch.

Every exclusion is recorded with a code and a reason, and the full list is
exported alongside the plan.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..campaign import Campaign, SelectionSettings
from ..data.loaders import CandidateSet, MeasurementSet
from ..data.validation import Exclusion, ExclusionCode, exclusions_frame
from ..sequences import substitutions


@dataclass
class FilterResult:
    """Eligible candidates and the reasons the others were set aside."""

    eligible: pd.DataFrame
    exclusions: list[Exclusion] = field(default_factory=list)
    settings_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def n_eligible(self) -> int:
        return int(len(self.eligible))

    @property
    def n_excluded(self) -> int:
        return len(self.exclusions)

    def exclusions_table(self) -> pd.DataFrame:
        return exclusions_frame(self.exclusions)

    def counts_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exclusion in self.exclusions:
            counts[exclusion.code] = counts.get(exclusion.code, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_eligible": self.n_eligible,
            "n_excluded": self.n_excluded,
            "counts_by_code": self.counts_by_code(),
            "settings_applied": dict(self.settings_applied),
        }


def filter_candidates(
    candidates: CandidateSet,
    campaign: Campaign,
    *,
    measurements: MeasurementSet | None = None,
    settings: SelectionSettings | None = None,
    restrict_to_candidate_ids: Iterable[str] | None = None,
    restriction_reason: str = "outside the evaluated region",
) -> FilterResult:
    """Apply every eligibility rule in a fixed order.

    The rules, in the order they are applied:

    1. substitutions at locked positions,
    2. more substitutions than ``max_substitutions`` allows,
    3. sequences already measured in this campaign (unless repeats are
       requested),
    4. an optional restriction to a named set of candidates, used when the
       researcher chooses to stay inside the evaluated region.
    """
    settings = settings or campaign.selection
    reference = campaign.reference_sequence
    locked = set(settings.locked_positions)
    measured_hashes = measurements.measured_hashes if measurements is not None else set()
    allowed_ids = set(restrict_to_candidate_ids) if restrict_to_candidate_ids is not None else None

    exclusions: list[Exclusion] = []
    keep_rows: list[int] = []

    for position, candidate in enumerate(candidates.frame.itertuples(index=False)):
        candidate_id = candidate.candidate_id
        subs = substitutions(reference, candidate.sequence)

        touched_locked = sorted({s.position for s in subs} & locked)
        if touched_locked:
            exclusions.append(
                Exclusion(
                    code=ExclusionCode.LOCKED_POSITION,
                    identifier=candidate_id,
                    detail=(
                        "substitutes locked position(s) "
                        + ", ".join(str(p) for p in touched_locked)
                    ),
                    severity="excluded",
                    row=int(getattr(candidate, "source_row", position + 2)),
                )
            )
            continue

        if settings.max_substitutions is not None and len(subs) > settings.max_substitutions:
            exclusions.append(
                Exclusion(
                    code=ExclusionCode.TOO_MANY_SUBSTITUTIONS,
                    identifier=candidate_id,
                    detail=(f"{len(subs)} substitutions, limit is {settings.max_substitutions}"),
                    severity="excluded",
                    row=int(getattr(candidate, "source_row", position + 2)),
                )
            )
            continue

        if candidate.seq_hash in measured_hashes and not settings.allow_measured_repeats:
            exclusions.append(
                Exclusion(
                    code=ExclusionCode.ALREADY_MEASURED,
                    identifier=candidate_id,
                    detail=(
                        "this sequence has already been measured in this campaign; set"
                        " allow_measured_repeats to order it again deliberately"
                    ),
                    severity="excluded",
                    row=int(getattr(candidate, "source_row", position + 2)),
                )
            )
            continue

        if allowed_ids is not None and candidate_id not in allowed_ids:
            exclusions.append(
                Exclusion(
                    code="restricted",
                    identifier=candidate_id,
                    detail=restriction_reason,
                    severity="excluded",
                    row=int(getattr(candidate, "source_row", position + 2)),
                )
            )
            continue

        keep_rows.append(position)

    eligible = candidates.frame.iloc[keep_rows].reset_index(drop=True)
    return FilterResult(
        eligible=eligible,
        exclusions=exclusions,
        settings_applied={
            "locked_positions": list(settings.locked_positions),
            "max_substitutions": settings.max_substitutions,
            "allow_measured_repeats": settings.allow_measured_repeats,
            "restricted_to_evaluated_range": allowed_ids is not None,
        },
    )

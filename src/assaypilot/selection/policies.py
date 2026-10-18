"""Two transparent selection policies.

``top_ranked``
    Take the highest-ranked eligible candidates. Simple, and the default.

``diversity``
    Take the highest-ranked candidate, then suppress every remaining candidate
    closer than a visible Hamming-distance threshold, then continue down the
    ranking. Useful when the pool contains clusters of near-identical
    sequences and spending the whole batch inside one cluster would waste it.

The diversity policy is an engineering option, not an assumed improvement:
whether spreading a batch out helps depends on the assay and the pool, and
``assaypilot benchmark`` exists to test it against pure ranking rather than
to assume it.

If a policy cannot fill the requested batch, it reports the shortfall. It
never relaxes the constraint quietly to reach the requested count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..sequences import hamming_distance


@dataclass
class SelectionOutcome:
    """The chosen batch, the reason for each choice, and any shortfall."""

    selected: pd.DataFrame
    policy: str
    requested: int
    shortfall: int = 0
    shortfall_reason: str = ""
    notes: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return int(len(self.selected))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "requested": self.requested,
            "selected": self.n_selected,
            "shortfall": self.shortfall,
            "shortfall_reason": self.shortfall_reason,
            "notes": list(self.notes),
            "parameters": dict(self.parameters),
        }


class SelectionPolicy(ABC):
    """Chooses a batch from ranked, eligible candidates."""

    name: str = "policy"

    @abstractmethod
    def select(self, ranked: pd.DataFrame, batch_size: int) -> SelectionOutcome:
        """``ranked`` must be sorted best-first and contain ``candidate_id`` and ``sequence``."""

    def describe(self) -> str:
        return self.name


class TopRankedPolicy(SelectionPolicy):
    """Take the top ``batch_size`` eligible candidates by predicted desirability."""

    name = "top_ranked"

    def select(self, ranked: pd.DataFrame, batch_size: int) -> SelectionOutcome:
        chosen = ranked.head(batch_size).copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        chosen["selection_reason"] = [
            f"rank {i} of {len(ranked)} eligible candidates by predicted desirability"
            for i in chosen["selection_rank"]
        ]
        shortfall = max(0, batch_size - len(chosen))
        return SelectionOutcome(
            selected=chosen,
            policy=self.name,
            requested=batch_size,
            shortfall=shortfall,
            shortfall_reason=(
                f"only {len(ranked)} eligible candidates remained after filtering"
                if shortfall
                else ""
            ),
            parameters={},
        )

    def describe(self) -> str:
        return "highest-ranked eligible candidates"


class DiversityPolicy(SelectionPolicy):
    """Rank-ordered selection with a minimum Hamming distance between picks.

    Parameters
    ----------
    min_distance:
        After a candidate is selected, every remaining candidate within this
        Hamming distance of it is suppressed for the rest of the batch.
    """

    name = "diversity"

    def __init__(self, min_distance: int) -> None:
        if min_distance < 1:
            raise ValueError("min_distance must be at least 1")
        self.min_distance = int(min_distance)

    def select(self, ranked: pd.DataFrame, batch_size: int) -> SelectionOutcome:
        chosen_rows: list[int] = []
        chosen_sequences: list[str] = []
        suppressed: list[tuple[str, str]] = []
        reasons: list[str] = []

        sequences = ranked["sequence"].tolist()
        ids = ranked["candidate_id"].tolist()

        for position, sequence in enumerate(sequences):
            if len(chosen_rows) >= batch_size:
                break
            blocking = None
            for picked_index, picked_sequence in enumerate(chosen_sequences):
                if hamming_distance(sequence, picked_sequence) < self.min_distance:
                    blocking = ids[chosen_rows[picked_index]]
                    break
            if blocking is not None:
                suppressed.append((ids[position], blocking))
                continue
            chosen_rows.append(position)
            chosen_sequences.append(sequence)
            reasons.append(
                f"rank {position + 1} of {len(ranked)} eligible candidates, and at least"
                f" {self.min_distance} substitutions from every candidate already chosen"
            )

        chosen = ranked.iloc[chosen_rows].copy()
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        chosen["selection_reason"] = reasons

        shortfall = max(0, batch_size - len(chosen))
        notes: list[str] = []
        if suppressed:
            notes.append(
                f"{len(suppressed)} candidate(s) were suppressed for being closer than"
                f" {self.min_distance} substitutions to an already selected candidate."
            )
        reason = ""
        if shortfall:
            reason = (
                f"the diversity constraint (minimum Hamming distance {self.min_distance})"
                f" left only {len(chosen)} candidates that satisfy it. The constraint was"
                " not relaxed. Lower the distance, enlarge the pool, or accept the smaller"
                " batch."
            )

        return SelectionOutcome(
            selected=chosen,
            policy=self.name,
            requested=batch_size,
            shortfall=shortfall,
            shortfall_reason=reason,
            notes=notes,
            parameters={"min_distance": self.min_distance, "n_suppressed": len(suppressed)},
        )

    def describe(self) -> str:
        return (
            f"highest-ranked candidates, keeping at least {self.min_distance} substitutions"
            " between any two chosen"
        )


def build_policy(name: str, *, min_distance: int | None = None) -> SelectionPolicy:
    """Create a policy by name (``"top_ranked"`` or ``"diversity"``)."""
    if name == "top_ranked":
        return TopRankedPolicy()
    if name == "diversity":
        if min_distance is None:
            raise ValueError(
                "the diversity policy needs a minimum Hamming distance"
                " (selection.diversity_min_distance)"
            )
        return DiversityPolicy(min_distance)
    raise ValueError(f"unknown selection policy {name!r}; use 'top_ranked' or 'diversity'")

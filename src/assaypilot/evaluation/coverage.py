"""Does the evidence cover the candidates you are about to order?

A method can score well on held-out measurements and still be unsupported for
a particular candidate: the evaluation may have tested variants one or two
substitutions from the reference while the pool contains variants five away,
or carrying substitutions that never appeared in any measurement.

This module quantifies that gap per candidate and flags the candidates lying
outside the region the evaluation actually covered. The researcher can then
restrict selection to the evaluated range rather than trusting a prediction
that nothing supports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.loaders import CandidateSet, MeasurementSet
from ..sequences import min_distance_to_set, substitutions


@dataclass
class CoverageReport:
    """Per-candidate distance and novelty, plus the pool-level summary."""

    frame: pd.DataFrame
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_outside(self) -> int:
        if self.frame.empty:
            return 0
        return int((~self.frame["within_evaluated_range"]).sum())

    def to_dict(self) -> dict[str, Any]:
        return {"summary": dict(self.summary), "warnings": list(self.warnings)}


def _substitution_set(reference: str, sequences: Sequence[str]) -> set[tuple[int, str]]:
    return {
        (s.position, s.mutant) for sequence in sequences for s in substitutions(reference, sequence)
    }


def assess_coverage(
    candidates: CandidateSet,
    measurements: MeasurementSet,
    *,
    evaluated_distances: Sequence[int] | None = None,
) -> CoverageReport:
    """Compare the candidate pool with the variants the evaluation actually covered.

    Parameters
    ----------
    evaluated_distances:
        How far each held-out sequence was from the training set **of its own
        split** (see
        :func:`~assaypilot.pipeline._evaluated_distances`). The largest of
        these is the furthest the evidence reaches. Computing the distances
        against a pooled training set would be wrong: with more than one split
        the same sequence is held out in one and trained on in another, so its
        distance would come out as zero and the evidence would look far
        narrower than it is.

    Candidate distances and substitution novelty are measured against **every**
    measured sequence, because that is what the final model is fitted on and
    what "never appears in the measurements" means to the reader.
    """
    reference = candidates.campaign.reference_sequence
    measured = measurements.frame["sequence"].tolist()
    known_substitutions = _substitution_set(reference, measured)
    training = measured

    distances_array = np.asarray(list(evaluated_distances or []), dtype=float)
    max_evaluated = float(distances_array.max()) if distances_array.size else float("nan")

    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.frame.iterrows():
        sequence = candidate["sequence"]
        subs = substitutions(reference, sequence)
        unseen = [s for s in subs if (s.position, s.mutant) not in known_substitutions]
        distance = min_distance_to_set(sequence, training) if training else len(sequence)
        within = bool(np.isfinite(max_evaluated) and distance <= max_evaluated)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "n_substitutions": int(candidate["n_substitutions"]),
                "min_distance_to_measured": int(distance),
                "n_unseen_substitutions": len(unseen),
                "unseen_substitutions": ":".join(str(s) for s in unseen),
                "within_evaluated_range": within,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "candidate_id",
            "n_substitutions",
            "min_distance_to_measured",
            "n_unseen_substitutions",
            "unseen_substitutions",
            "within_evaluated_range",
        ],
    )

    warnings: list[str] = []
    summary: dict[str, Any] = {
        "n_candidates": int(len(frame)),
        "max_evaluated_distance": None if np.isnan(max_evaluated) else max_evaluated,
        "evaluated_distance_median": (
            float(np.median(distances_array)) if distances_array.size else None
        ),
    }

    if not frame.empty:
        distances = frame["min_distance_to_measured"].to_numpy()
        summary.update(
            {
                "candidate_distance_median": float(np.median(distances)),
                "candidate_distance_max": int(distances.max()),
                "n_outside_evaluated_range": int((~frame["within_evaluated_range"]).sum()),
                "fraction_with_unseen_substitutions": float(
                    (frame["n_unseen_substitutions"] > 0).mean()
                ),
            }
        )

        if not np.isfinite(max_evaluated):
            warnings.append(
                "No held-out evaluation was available, so no candidate can be described as"
                " lying inside an evaluated region. Treat all predictions as exploratory."
            )
        else:
            outside = summary["n_outside_evaluated_range"]
            if outside:
                warnings.append(
                    f"{outside} of {len(frame)} candidates are further from the measured"
                    f" variants than anything the evaluation covered (more than"
                    f" {max_evaluated:.0f} substitutions away). The reported performance does"
                    " not extend to them; you can restrict selection to the evaluated range."
                )
        novel = float((frame["n_unseen_substitutions"] > 0).mean())
        if novel > 0.5:
            warnings.append(
                f"{novel:.0%} of candidates carry at least one substitution that never appears"
                " in the measurements. The simple baseline cannot represent those"
                " substitutions at all, and the other methods are extrapolating."
            )

    return CoverageReport(frame=frame, summary=summary, warnings=warnings)

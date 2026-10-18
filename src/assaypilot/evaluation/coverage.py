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
    evaluated_sequences: Sequence[str] | None = None,
    training_sequences: Sequence[str] | None = None,
) -> CoverageReport:
    """Compare the candidate pool with the variants the evaluation actually covered.

    Parameters
    ----------
    evaluated_sequences:
        The held-out sequences the reported metrics were computed on. Their
        distance from the training set defines the region the evidence covers.
    training_sequences:
        The sequences the final model is fitted on. Defaults to every measured
        sequence.
    """
    reference = candidates.campaign.reference_sequence
    measured = measurements.frame["sequence"].tolist()
    training = list(training_sequences) if training_sequences is not None else measured
    known_substitutions = _substitution_set(reference, training)

    if evaluated_sequences:
        evaluated_distances = np.array(
            [min_distance_to_set(seq, training) for seq in evaluated_sequences], dtype=float
        )
    else:
        evaluated_distances = np.array([], dtype=float)

    max_evaluated = float(evaluated_distances.max()) if evaluated_distances.size else float("nan")

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
            float(np.median(evaluated_distances)) if evaluated_distances.size else None
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

"""Example data: a clearly-labelled simulation, and readers for public datasets.

Two very different things live here, and the distinction matters:

:func:`simulate_campaign`
    **Synthetic data.** A reproducible toy campaign with a known generative
    model (additive site effects, a little pairwise epistasis, measurement
    noise). It exists so the workflow can be demonstrated and tested without
    downloading anything. No number produced from it says anything about a
    real protein, and every artifact generated from it is labelled
    ``simulated``.

:func:`load_proteingym_assay` and :func:`load_flip_split`
    **Real public measurements**, read from files the user downloads
    themselves. AssayPilot does not redistribute these datasets; the scripts
    in ``scripts/`` fetch them and ``docs/benchmarks.md`` records which
    versions were used.

A note that applies to every public dataset used here: published assay
sequences may overlap the pretraining data of protein language models, so no
result on them should be described as evaluation on sequences the model has
never seen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .campaign import Campaign, SelectionSettings
from .errors import InputError
from .sequences import AMINO_ACIDS, normalise_sequence, validate_sequence

#: A short, synthetic reference sequence. It is not a real protein.
SIMULATED_REFERENCE = (
    "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAG"
    "WILGNPECESLSTASSWSYIVETSSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAA"
)


@dataclass
class SimulatedCampaign:
    """A generated campaign with its measurements, candidates, and ground truth."""

    campaign: Campaign
    measurements: pd.DataFrame
    candidates: pd.DataFrame
    truth: pd.DataFrame
    description: str

    def write(self, directory: str | Path) -> dict[str, Path]:
        """Write the three CSV files plus the campaign definition."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "campaign": self.campaign.to_json(directory / "campaign.json"),
            "measurements": directory / "measurements.csv",
            "candidates": directory / "candidates.csv",
            "truth": directory / "ground_truth.csv",
        }
        self.measurements.to_csv(paths["measurements"], index=False)
        self.candidates.to_csv(paths["candidates"], index=False)
        self.truth.to_csv(paths["truth"], index=False)
        (directory / "README.md").write_text(self.description, encoding="utf-8")
        return paths


def _random_variant(
    rng: np.random.Generator,
    reference: str,
    n_substitutions: int,
    allowed_positions: Sequence[int],
) -> tuple[str, tuple[int, ...]]:
    """Draw a variant that substitutes ``n_substitutions`` of the allowed positions.

    ``allowed_positions`` are 0-based. Restricting them is what makes the
    simulation resemble a real designed library, where a campaign returns to
    the same handful of residues rather than scattering substitutions over the
    whole protein.
    """
    n_substitutions = min(n_substitutions, len(allowed_positions))
    positions = rng.choice(allowed_positions, size=n_substitutions, replace=False)
    residues = list(reference)
    for position in positions:
        current = reference[position]
        choices = [aa for aa in AMINO_ACIDS if aa != current]
        residues[position] = str(rng.choice(choices))
    return "".join(residues), tuple(sorted(int(p) + 1 for p in positions))


def simulate_campaign(
    *,
    reference: str = SIMULATED_REFERENCE,
    n_measured: int = 150,
    n_candidates: int = 1000,
    n_rounds: int = 3,
    n_variable_positions: int = 24,
    novel_position_fraction: float = 0.1,
    substitutions_measured: tuple[int, ...] = (1, 1, 2, 2, 3),
    substitutions_candidates: tuple[int, ...] = (2, 3, 3, 4),
    replicate_fraction: float = 0.15,
    noise_sd: float = 0.35,
    epistasis_strength: float = 0.4,
    campaign_id: str = "simulated-enzyme",
    seed: int = 7,
) -> SimulatedCampaign:
    """Generate a reproducible synthetic campaign.

    The generative model is deliberately simple and stated in full in the
    returned ``description``: each position/residue pair has a fixed additive
    effect, a small number of position pairs interact, and each measurement
    carries Gaussian noise. Candidates are drawn with *more* substitutions
    than the measured variants, which is the realistic and difficult case —
    the evaluation then has to extrapolate.

    Returns measurements with ``round_id`` labels and replicates, a candidate
    pool, and the noise-free ground truth for every sequence involved (used by
    the replay harness, never by the planner).
    """
    reference = validate_sequence(reference, label="simulated reference")
    rng = np.random.default_rng(seed)

    # A designed library: the campaign varies a fixed panel of positions, as a
    # real engineering effort would, rather than scattering substitutions over
    # the whole protein. A minority of candidates reach beyond that panel, which
    # is what exercises the coverage warnings.
    library_positions = sorted(
        int(p) for p in rng.choice(len(reference), size=n_variable_positions, replace=False)
    )
    novel_positions = sorted(
        int(p)
        for p in rng.choice(
            [i for i in range(len(reference)) if i not in set(library_positions)],
            size=max(2, n_variable_positions // 4),
            replace=False,
        )
    )

    # Additive effects for every (position, residue) pair, plus a few interacting pairs.
    site_effects = rng.normal(0.0, 1.0, size=(len(reference), len(AMINO_ACIDS)))
    aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    n_interactions = max(1, n_variable_positions // 3)
    interaction_pairs = [
        (int(a), int(b))
        for a, b in rng.choice(library_positions, size=(n_interactions, 2), replace=True)
        if a != b
    ]
    interaction_weights = rng.normal(0.0, epistasis_strength, size=len(interaction_pairs))

    def true_value(sequence: str) -> float:
        total = 0.0
        changed = [i for i, (r, m) in enumerate(zip(reference, sequence, strict=True)) if r != m]
        for position in changed:
            total += float(site_effects[position, aa_index[sequence[position]]])
        changed_set = set(changed)
        for weight, (a, b) in zip(interaction_weights, interaction_pairs, strict=True):
            if a in changed_set and b in changed_set:
                total += float(weight)
        return total

    # ---- measured variants, spread across rounds
    measured_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_round = max(1, n_measured // max(n_rounds, 1))

    variant_number = 0
    for round_index in range(1, n_rounds + 1):
        # Later rounds explore slightly more substitutions, as a real campaign would.
        pool = substitutions_measured + (round_index,)
        for _ in range(per_round):
            for _attempt in range(50):
                n_subs = int(rng.choice(pool))
                sequence, _positions = _random_variant(rng, reference, n_subs, library_positions)
                if sequence not in seen:
                    break
            else:  # pragma: no cover - extremely unlikely
                continue
            seen.add(sequence)
            variant_number += 1
            variant_id = f"var{variant_number:04d}"
            signal = true_value(sequence)
            truth_rows.append(
                {"variant_id": variant_id, "sequence": sequence, "true_value": signal}
            )
            n_replicates = 2 if rng.random() < replicate_fraction else 1
            for replicate in range(1, n_replicates + 1):
                measured_rows.append(
                    {
                        "variant_id": variant_id,
                        "sequence": sequence,
                        "value": signal + rng.normal(0.0, noise_sd),
                        "condition_id": "37C_pH7.4",
                        "round_id": f"R{round_index}",
                        "replicate_id": f"rep{replicate}",
                        "plate_id": f"P{round_index}{(variant_number % 2) + 1}",
                    }
                )

    # ---- candidate pool, deliberately further from the reference
    candidate_rows: list[dict[str, Any]] = []
    for index in range(1, n_candidates + 1):
        reaches_beyond = rng.random() < novel_position_fraction
        positions_for_candidate = (
            library_positions + novel_positions if reaches_beyond else library_positions
        )
        for _attempt in range(50):
            n_subs = int(rng.choice(substitutions_candidates))
            sequence, _positions = _random_variant(rng, reference, n_subs, positions_for_candidate)
            if sequence not in seen:
                break
        else:  # pragma: no cover
            continue
        seen.add(sequence)
        candidate_id = f"cand{index:04d}"
        candidate_rows.append({"candidate_id": candidate_id, "sequence": sequence})
        truth_rows.append(
            {
                "variant_id": candidate_id,
                "sequence": sequence,
                "true_value": true_value(sequence),
            }
        )

    measurements = pd.DataFrame(measured_rows)
    candidates = pd.DataFrame(candidate_rows)
    truth = pd.DataFrame(truth_rows)

    threshold = float(np.quantile(measurements["value"], 0.9))
    campaign = Campaign(
        campaign_id=campaign_id,
        reference_sequence=reference,
        assay_name="simulated activity (arbitrary units)",
        assay_units="simulated AU",
        objective="maximize",
        condition_id="37C_pH7.4",
        description=(
            "SIMULATED campaign generated by assaypilot.datasets.simulate_campaign."
            " Not a real protein and not real measurements."
        ),
        success_threshold=threshold,
        selection=SelectionSettings(batch_size=24, max_substitutions=5),
    )

    description = f"""# Simulated example campaign

**This data is synthetic.** It was generated by
`assaypilot.datasets.simulate_campaign(seed={seed})` and describes no real protein,
no real assay, and no real measurement. It exists so that the AssayPilot workflow can be
demonstrated and tested without downloading anything.

Generative model:

- the campaign varies a designed panel of {n_variable_positions} positions, as a real
  engineering effort would; about {novel_position_fraction:.0%} of candidates also reach
  positions outside that panel, which is what exercises the coverage warnings;
- every (position, residue) pair has a fixed additive effect drawn from N(0, 1);
- {len(interaction_pairs)} pairs of positions interact, with weights drawn from
  N(0, {epistasis_strength}), so the data is not purely additive;
- each measurement adds Gaussian noise with standard deviation {noise_sd};
- about {replicate_fraction:.0%} of variants were measured twice, so replicate handling is
  exercised;
- measured variants carry {min(substitutions_measured)}-{max(substitutions_measured) + n_rounds}
  substitutions; candidates carry {min(substitutions_candidates)}-{max(substitutions_candidates)},
  which makes the candidate pool genuinely harder than the training data.

Files: `measurements.csv`, `candidates.csv`, `campaign.json`, and `ground_truth.csv`
(noise-free values for replay only — the planner never reads it).
"""

    return SimulatedCampaign(
        campaign=campaign,
        measurements=measurements,
        candidates=candidates,
        truth=truth,
        description=description,
    )


def load_proteingym_assay(
    path: str | Path,
    *,
    campaign_id: str | None = None,
    assay_name: str = "DMS score",
    assay_units: str = "score (dataset units)",
    condition_id: str = "published",
    objective: str = "maximize",
    reference_sequence: str | None = None,
) -> tuple[Campaign, pd.DataFrame]:
    """Read a ProteinGym-style substitution DMS file into AssayPilot inputs.

    Expected columns are those ProteinGym uses for substitution assays:
    ``mutant`` (for example ``A12G:L30P``), ``mutated_sequence``, and
    ``DMS_score``. When ``reference_sequence`` is not supplied it is
    reconstructed from a single-substitution row, which is the only case where
    the wild-type residues in the notation determine it unambiguously.

    AssayPilot does not ship these files. Download them from proteingym.org
    and record the release you used; ``docs/benchmarks.md`` explains how the
    bundled comparison figures were produced.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required - set(frame.columns)
    if missing:
        raise InputError(
            f"{path.name} does not look like a ProteinGym substitution file:"
            f" missing column(s) {', '.join(sorted(missing))}"
        )

    if reference_sequence is None:
        single = frame[~frame["mutant"].astype(str).str.contains(":")]
        if single.empty:
            raise InputError(
                "cannot reconstruct the reference sequence from this file: it contains no"
                " single-substitution rows. Supply reference_sequence explicitly."
            )
        row = single.iloc[0]
        notation = str(row["mutant"])
        position = int(notation[1:-1])
        wildtype = notation[0]
        sequence = list(normalise_sequence(str(row["mutated_sequence"])))
        sequence[position - 1] = wildtype
        reference_sequence = "".join(sequence)

    campaign = Campaign(
        campaign_id=campaign_id or path.stem.replace("_", "-")[:60],
        reference_sequence=reference_sequence,
        assay_name=assay_name,
        assay_units=assay_units,
        objective=objective,  # type: ignore[arg-type]
        condition_id=condition_id,
        description=f"Public assay data read from {path.name}.",
    )

    measurements = pd.DataFrame(
        {
            "variant_id": frame["mutant"].astype(str),
            "sequence": frame["mutated_sequence"].astype(str),
            "value": frame["DMS_score"].astype(float),
            "condition_id": condition_id,
        }
    )
    return campaign, measurements


def load_flip_split(
    sequences_path: str | Path,
    *,
    split_column: str = "set",
    value_column: str = "target",
) -> pd.DataFrame:
    """Read a FLIP split file, keeping its own train/test assignment intact.

    FLIP ships biologically motivated partitions. When a FLIP split is used,
    AssayPilot keeps the published assignment rather than inventing its own,
    so results stay comparable with the published task.
    """
    frame = pd.read_csv(sequences_path)
    for column in (split_column, value_column, "sequence"):
        if column not in frame.columns:
            raise InputError(
                f"{Path(sequences_path).name} is missing the {column!r} column expected in a"
                " FLIP split file"
            )
    return frame

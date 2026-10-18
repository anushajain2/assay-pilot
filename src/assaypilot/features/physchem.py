"""A dependency-free physicochemical representation.

This backend exists for one reason: the whole AssayPilot workflow — unit tests,
continuous integration, the bundled demo, and a first look on a laptop with no
GPU — must run without downloading model weights.

It is **not** a protein language model and makes no attempt to be one. It
encodes seven standard residue property scales, pooled globally and in
positional bins, so that a substitution at residue 40 is represented
differently from the same substitution at residue 300. Everywhere this backend
is used, reports label it ``physicochemical properties (not a protein language
model)`` so that a reader never mistakes its results for ESM-2 results.

Property scales are published, widely used residue indices; see
``docs/representations.md`` for the sources and the exact values.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..sequences import AMINO_ACIDS
from .base import EmbeddingBackend

#: Seven residue property scales, in the column order used by the encoder.
PROPERTY_NAMES: tuple[str, ...] = (
    "hydropathy",  # Kyte & Doolittle
    "volume",  # residue volume, A^3
    "charge",  # formal charge at pH 7
    "polarity",  # Grantham polarity
    "isoelectric_point",
    "helix_propensity",  # Pace & Scholtz, kcal/mol (lower = more helix-favourable)
    "aromaticity",  # 1 for F, W, Y, H; 0 otherwise
)

# One row per amino acid, columns in PROPERTY_NAMES order.
_PROPERTIES: dict[str, tuple[float, ...]] = {
    "A": (1.80, 88.6, 0.0, 8.1, 6.00, 0.00, 0.0),
    "C": (2.50, 108.5, 0.0, 5.5, 5.07, 0.68, 0.0),
    "D": (-3.50, 111.1, -1.0, 13.0, 2.77, 0.69, 0.0),
    "E": (-3.50, 138.4, -1.0, 12.3, 3.22, 0.40, 0.0),
    "F": (2.80, 189.9, 0.0, 5.2, 5.48, 0.54, 1.0),
    "G": (-0.40, 60.1, 0.0, 9.0, 5.97, 1.00, 0.0),
    "H": (-3.20, 153.2, 0.1, 10.4, 7.59, 0.61, 1.0),
    "I": (4.50, 166.7, 0.0, 5.2, 6.02, 0.41, 0.0),
    "K": (-3.90, 168.6, 1.0, 11.3, 9.74, 0.26, 0.0),
    "L": (3.80, 166.7, 0.0, 4.9, 5.98, 0.21, 0.0),
    "M": (1.90, 162.9, 0.0, 5.7, 5.74, 0.24, 0.0),
    "N": (-3.50, 114.1, 0.0, 11.6, 5.41, 0.65, 0.0),
    "P": (-1.60, 112.7, 0.0, 8.0, 6.30, 3.16, 0.0),
    "Q": (-3.50, 143.8, 0.0, 10.5, 5.65, 0.39, 0.0),
    "R": (-4.50, 173.4, 1.0, 10.5, 10.76, 0.21, 0.0),
    "S": (-0.80, 89.0, 0.0, 9.2, 5.68, 0.50, 0.0),
    "T": (-0.70, 116.1, 0.0, 8.6, 5.60, 0.66, 0.0),
    "V": (4.20, 140.0, 0.0, 5.9, 5.96, 0.61, 0.0),
    "W": (-0.90, 227.8, 0.0, 5.4, 5.89, 0.49, 1.0),
    "Y": (-1.30, 193.6, 0.0, 6.2, 5.66, 0.53, 1.0),
}

_MATRIX = np.array([_PROPERTIES[aa] for aa in AMINO_ACIDS], dtype=np.float32)
_MEAN = _MATRIX.mean(axis=0)
_STD = _MATRIX.std(axis=0)
_STD[_STD == 0] = 1.0
_SCALED = (_MATRIX - _MEAN) / _STD
_LOOKUP: dict[str, np.ndarray] = {aa: _SCALED[i] for i, aa in enumerate(AMINO_ACIDS)}


class PhysicochemicalBackend(EmbeddingBackend):
    """Positionally binned physicochemical property features.

    Parameters
    ----------
    n_bins:
        Number of contiguous positional bins the sequence is divided into. Each
        bin contributes the mean property vector of its residues, which makes
        the representation sensitive to *where* a substitution occurred while
        keeping the dimension independent of sequence length.
    """

    name = "physchem"

    def __init__(self, *, n_bins: int = 16) -> None:
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.n_bins = n_bins

    @property
    def description(self) -> str:
        return (
            f"physicochemical properties in {self.n_bins} positional bins"
            " (not a protein language model)"
        )

    @property
    def n_features(self) -> int:
        return len(PROPERTY_NAMES) * (self.n_bins + 1)

    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(sequences), self.n_features), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            residues = np.vstack([_LOOKUP[aa] for aa in sequence])
            edges = np.linspace(0, len(sequence), self.n_bins + 1).astype(int)
            parts = [residues.mean(axis=0)]
            for start, stop in zip(edges[:-1], edges[1:], strict=True):
                if stop > start:
                    parts.append(residues[start:stop].mean(axis=0))
                else:  # bin empty for very short sequences
                    parts.append(np.zeros(len(PROPERTY_NAMES), dtype=np.float32))
            out[row] = np.concatenate(parts)
        return out

    def fingerprint(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "version": 1,
            "n_bins": self.n_bins,
            "properties": list(PROPERTY_NAMES),
            "n_features": self.n_features,
            "is_language_model": False,
        }

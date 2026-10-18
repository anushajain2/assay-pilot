"""One-hot encoding of substitutions, used by the simple baseline.

The encoder learns its vocabulary of ``(position, mutant residue)`` pairs from
the sequences it is *fitted* on. That matters for evaluation honesty: when the
encoder sits inside a scikit-learn pipeline, the vocabulary is built from the
training fold only, so a held-out variant carrying an unseen substitution is
represented as all-zeros rather than leaking its existence into training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..sequences import substitutions


class SubstitutionOneHotEncoder(BaseEstimator, TransformerMixin):
    """Binary indicators for each substitution seen during fitting.

    Parameters
    ----------
    reference:
        The campaign reference sequence. Substitutions are expressed relative
        to it.
    include_count:
        Append the total number of substitutions as an extra feature. This
        gives the baseline a cheap way to represent a general mutational-load
        effect.
    min_frequency:
        Drop substitutions seen fewer than this many times during fitting.
    """

    def __init__(
        self,
        reference: str,
        *,
        include_count: bool = True,
        min_frequency: int = 1,
    ) -> None:
        self.reference = reference
        self.include_count = include_count
        self.min_frequency = min_frequency

    def fit(self, X: Sequence[str], y=None) -> SubstitutionOneHotEncoder:  # noqa: N803
        counts: dict[tuple[int, str], int] = {}
        for sequence in X:
            for sub in substitutions(self.reference, sequence):
                key = (sub.position, sub.mutant)
                counts[key] = counts.get(key, 0) + 1
        kept = sorted(k for k, c in counts.items() if c >= self.min_frequency)
        self.vocabulary_: dict[tuple[int, str], int] = {key: i for i, key in enumerate(kept)}
        self.feature_names_: list[str] = [
            f"{self.reference[pos - 1]}{pos}{mut}" for pos, mut in kept
        ]
        if self.include_count:
            self.feature_names_.append("n_substitutions")
        self.n_features_out_ = len(self.vocabulary_) + (1 if self.include_count else 0)
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:  # noqa: N803
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("SubstitutionOneHotEncoder.transform called before fit")
        X = list(X)
        matrix = np.zeros((len(X), max(self.n_features_out_, 1)), dtype=np.float32)
        for row, sequence in enumerate(X):
            subs = substitutions(self.reference, sequence)
            for sub in subs:
                index = self.vocabulary_.get((sub.position, sub.mutant))
                if index is not None:
                    matrix[row, index] = 1.0
            if self.include_count:
                matrix[row, -1] = float(len(subs))
        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.feature_names_, dtype=object)

    def unseen_substitution_fraction(self, X: Sequence[str]) -> float:
        """Fraction of substitutions in ``X`` that the fitted vocabulary cannot represent.

        Reported alongside baseline results: a high value means the baseline is
        effectively predicting the mean for those variants.
        """
        total = 0
        unseen = 0
        for sequence in X:
            for sub in substitutions(self.reference, sequence):
                total += 1
                if (sub.position, sub.mutant) not in self.vocabulary_:
                    unseen += 1
        return unseen / total if total else 0.0

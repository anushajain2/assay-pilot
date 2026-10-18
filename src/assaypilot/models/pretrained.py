"""The pretrained, label-free method: ESM-2 masked-marginal mutation scores.

This method never sees the assay labels. That is what makes it informative:
if it ranks held-out variants as well as the assay-trained model, the labels
were adding little; if it ranks them worse, the training is doing work.

Its known limits travel with it. Scores for multi-substitution variants are a
**sum of single-site terms**, which assumes the substitutions act
independently. The method also produces a score on an arbitrary scale, not a
predicted assay value, so AssayPilot exports it as ``score`` and leaves
``predicted_value`` empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import BackendUnavailableError
from ..features.esm import DEFAULT_CHECKPOINT, MaskedMarginalScorer, esm_available
from ..sequences import substitutions
from .base import Method, MethodInfo


class PretrainedScoreMethod(Method):
    """ESM-2 masked-marginal score, used directly as a ranking.

    Parameters
    ----------
    reference:
        Campaign reference sequence; substitutions are scored relative to it.
    checkpoint, revision, device:
        Passed to :class:`~assaypilot.features.esm.MaskedMarginalScorer`. Pin
        ``revision`` to make the result reproducible.
    """

    def __init__(
        self,
        reference: str,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        revision: str | None = None,
        device: str | None = None,
        scorer: MaskedMarginalScorer | None = None,
    ) -> None:
        self.reference = reference
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device
        self._scorer = scorer
        self._fitted = False

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key="pretrained_esm_masked_marginal",
            name="Pretrained score (ESM-2 masked marginals)",
            kind="pretrained",
            description=(
                "Label-free mutation score from a pretrained protein language model:"
                " log p(mutant) - log p(wild type) at each masked position."
            ),
            uses_labels=False,
            complexity=2,
            caveats=(
                "Scores for variants with more than one substitution are a sum of single-site"
                " terms and therefore cannot represent epistasis.",
                "The output is a model score on an arbitrary scale, not a predicted assay value.",
                "Sequence plausibility under a language model is not a measurement of the"
                " assayed function.",
            ),
            details={"checkpoint": self.checkpoint, "revision": self.revision or "unpinned"},
        )

    def is_available(self) -> tuple[bool, str]:
        if self._scorer is not None:
            return True, ""
        if not esm_available():
            return False, (
                "torch and transformers are not installed, so the pretrained ESM-2 score"
                " cannot be computed. Install with 'pip install \"assaypilot[esm]\"'."
            )
        return True, ""

    @property
    def scorer(self) -> MaskedMarginalScorer:
        if self._scorer is None:
            available, reason = self.is_available()
            if not available:
                raise BackendUnavailableError(reason)
            self._scorer = MaskedMarginalScorer(
                self.reference,
                checkpoint=self.checkpoint,
                revision=self.revision,
                device=self.device,
            )
        return self._scorer

    def clone(self) -> PretrainedScoreMethod:
        # The scorer is stateless with respect to labels and caches only
        # reference log-probabilities, so it is safe (and much faster) to share
        # across cross-validation folds.
        return PretrainedScoreMethod(
            self.reference,
            checkpoint=self.checkpoint,
            revision=self.revision,
            device=self.device,
            scorer=self._scorer,
        )

    def fit(
        self, sequences: Sequence[str], y: np.ndarray, *, groups: np.ndarray | None = None
    ) -> PretrainedScoreMethod:
        """No-op: this method does not use labels. Present for interface symmetry."""
        del sequences, y, groups
        self._fitted = True
        return self

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        return self.scorer.score(list(sequences))

    def multi_substitution_fraction(self, sequences: Sequence[str]) -> float:
        """Share of sequences whose score relies on the additive approximation."""
        sequences = list(sequences)
        if not sequences:
            return 0.0
        multi = sum(1 for s in sequences if len(substitutions(self.reference, s)) > 1)
        return multi / len(sequences)

    def fit_details(self) -> dict[str, Any]:
        if self._scorer is None:
            return {"checkpoint": self.checkpoint, "revision": self.revision or "unpinned"}
        return self._scorer.fingerprint()

"""The common interface every prediction method implements.

Three rules hold for all methods:

1. **Predictions are desirability, not raw assay values.** Higher always means
   "better under this campaign's objective". The pipeline sign-flips the
   measurements for a minimisation objective before fitting, so ranking,
   metrics, and selection never need to know the direction.
2. **Fitting happens on training data only.** Any preprocessing that learns
   from data — one-hot vocabulary, scaling, dimensionality reduction,
   hyperparameters — happens inside :meth:`Method.fit`, so that a method
   placed inside a cross-validation loop cannot see its held-out examples.
3. **A method describes its own limits.** :attr:`MethodInfo.caveats` carries the
   sentences that must appear next to its numbers, for example that a
   multi-substitution pretrained score is an additive approximation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MethodKind = str  # "baseline" | "pretrained" | "assay_trained"


@dataclass(frozen=True)
class MethodInfo:
    """What a method is, in the words used in reports and manifests."""

    key: str
    name: str
    kind: MethodKind
    description: str
    uses_labels: bool
    complexity: int
    caveats: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "uses_labels": self.uses_labels,
            "complexity": self.complexity,
            "caveats": list(self.caveats),
            "details": dict(self.details),
        }


class Method(ABC):
    """Fit on sequences and desirability values; predict desirability."""

    @property
    @abstractmethod
    def info(self) -> MethodInfo:
        """Static description of the method, including its caveats."""

    @abstractmethod
    def fit(
        self, sequences: Sequence[str], y: np.ndarray, *, groups: np.ndarray | None = None
    ) -> Method:
        """Fit on training data. ``groups`` keeps replicates together in inner CV."""

    @abstractmethod
    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        """Predicted desirability, higher is better."""

    @abstractmethod
    def clone(self) -> Method:
        """A fresh, unfitted copy with the same configuration."""

    def is_available(self) -> tuple[bool, str]:
        """``(available, reason)``. An unavailable method is reported, not hidden."""
        return True, ""

    def fit_details(self) -> dict[str, Any]:
        """Chosen hyperparameters and other facts worth recording in the manifest."""
        return {}

    @property
    def key(self) -> str:
        return self.info.key

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} key={self.info.key!r}>"

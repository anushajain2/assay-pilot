"""The simple baseline: one-hot substitutions with ridge regression.

Its job is to answer a question the rest of the suite cannot answer about
itself: *does a protein-model representation add predictive value on this
assay, compared with a linear model over the substitutions themselves?*

Because that comparison is the point, the baseline is tuned as carefully as
the other methods — the ridge penalty is chosen by grouped inner
cross-validation on the training data only. A baseline that is deliberately
weak would make every other method look good.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline

from ..features.onehot import SubstitutionOneHotEncoder
from .base import Method, MethodInfo

DEFAULT_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _inner_cv(n_samples: int, groups: np.ndarray | None, max_splits: int = 5):
    """A grouped inner CV splitter sized for the data available.

    Returns ``None`` when there is too little data to split, in which case the
    caller falls back to a fixed default hyperparameter and says so.
    """
    if groups is not None:
        n_groups = len(np.unique(groups))
        n_splits = min(max_splits, n_groups)
        if n_splits < 2:
            return None
        return GroupKFold(n_splits=n_splits)
    n_splits = min(max_splits, n_samples)
    if n_splits < 2:
        return None
    return KFold(n_splits=n_splits, shuffle=True, random_state=0)


class OneHotRidgeBaseline(Method):
    """Ridge regression on binary substitution indicators."""

    def __init__(
        self,
        reference: str,
        *,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        include_count: bool = True,
    ) -> None:
        self.reference = reference
        self.alphas = tuple(alphas)
        self.include_count = include_count
        self._pipeline: Pipeline | None = None
        self._chosen: dict[str, Any] = {}

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key="baseline_onehot_ridge",
            name="Simple baseline (one-hot substitutions, ridge)",
            kind="baseline",
            description=(
                "Ridge regression on binary indicators of each substitution seen in training,"
                " plus the total number of substitutions."
            ),
            uses_labels=True,
            complexity=1,
            caveats=(
                "Cannot represent a substitution that never appears in the training data;"
                " such variants fall back towards the training mean.",
                "Linear and additive in substitutions, so it does not model epistasis.",
            ),
            details={"alphas": list(self.alphas), "include_count": self.include_count},
        )

    def clone(self) -> OneHotRidgeBaseline:
        return OneHotRidgeBaseline(
            self.reference, alphas=self.alphas, include_count=self.include_count
        )

    def fit(
        self, sequences: Sequence[str], y: np.ndarray, *, groups: np.ndarray | None = None
    ) -> OneHotRidgeBaseline:
        sequences = list(sequences)
        y = np.asarray(y, dtype=float)
        pipeline = Pipeline(
            [
                (
                    "encode",
                    SubstitutionOneHotEncoder(self.reference, include_count=self.include_count),
                ),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        cv = _inner_cv(len(sequences), groups)
        if cv is None or len(self.alphas) == 1:
            alpha = self.alphas[len(self.alphas) // 2]
            pipeline.set_params(ridge__alpha=alpha)
            pipeline.fit(sequences, y)
            self._chosen = {
                "alpha": alpha,
                "alpha_selection": "default (too little data for inner cross-validation)",
            }
        else:
            search = GridSearchCV(
                pipeline,
                {"ridge__alpha": list(self.alphas)},
                cv=cv,
                scoring="neg_mean_squared_error",
                error_score="raise",
            )
            search.fit(sequences, y, groups=groups)
            pipeline = search.best_estimator_
            self._chosen = {
                "alpha": float(search.best_params_["ridge__alpha"]),
                "alpha_selection": f"grouped inner cross-validation ({cv.get_n_splits()} folds)",
            }

        self._pipeline = pipeline
        encoder: SubstitutionOneHotEncoder = pipeline.named_steps["encode"]
        self._chosen["n_substitution_features"] = len(encoder.vocabulary_)
        return self

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("OneHotRidgeBaseline.predict called before fit")
        return np.asarray(self._pipeline.predict(list(sequences)), dtype=float)

    def unseen_substitution_fraction(self, sequences: Sequence[str]) -> float:
        """Share of substitutions in ``sequences`` the fitted encoder cannot represent."""
        if self._pipeline is None:
            raise RuntimeError("fit the baseline before asking about unseen substitutions")
        encoder: SubstitutionOneHotEncoder = self._pipeline.named_steps["encode"]
        return encoder.unseen_substitution_fraction(list(sequences))

    def fit_details(self) -> dict[str, Any]:
        return dict(self._chosen)

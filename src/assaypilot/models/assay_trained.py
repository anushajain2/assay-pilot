"""The assay-trained method: frozen representations with a regularized head.

The protein model is never fine-tuned. Sequences are embedded once — the
embedding depends only on the sequence, never on a label — and a small
pipeline of ``StandardScaler -> PCA -> Ridge`` is fitted on top.

Scaling, dimensionality reduction, and the ridge penalty are all fitted
**inside** the training folds of an inner grouped cross-validation. Doing the
scaling or the PCA once over the whole dataset would leak information from the
held-out examples into the fit; that is the preprocessing-leakage failure this
class is written to avoid, and ``tests/test_leakage.py`` checks it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.base import EmbeddingBackend, EmbeddingCache, embed_sequences
from .base import Method, MethodInfo
from .baseline import DEFAULT_ALPHAS, _inner_cv


class EmbeddingRidgeMethod(Method):
    """Frozen sequence embeddings with a ridge regression head.

    Parameters
    ----------
    backend:
        Representation backend (ESM-2, or the dependency-free physicochemical
        backend). Its :meth:`fingerprint` is recorded in the run manifest.
    cache:
        Optional embedding cache. Embeddings are label-independent, so caching
        them across folds and runs changes nothing about the evaluation.
    n_components:
        Candidate PCA sizes for the inner search. ``None`` in the list means
        "no dimensionality reduction".
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        cache: EmbeddingCache | None = None,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        n_components: Sequence[int | None] = (None, 16, 64),
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.alphas = tuple(alphas)
        self.n_components = tuple(n_components)
        self.batch_size = batch_size
        self._pipeline: Pipeline | None = None
        self._chosen: dict[str, Any] = {}

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key="assay_trained_embedding_ridge",
            name=f"Assay-trained model ({self.backend.name} embeddings + ridge)",
            kind="assay_trained",
            description=(
                f"Ridge regression on frozen {self.backend.description}."
                " Scaling, PCA, and the ridge penalty are fitted inside training folds only."
            ),
            uses_labels=True,
            complexity=3,
            caveats=(
                "Learns the relationship between a representation and this assay's measured"
                " outcome; it does not transfer to a different assay or condition.",
                "The protein model is frozen, so representation quality is fixed by the"
                " chosen checkpoint."
                if self.backend.is_language_model
                else "The representation is physicochemical, not a protein language model;"
                " results should not be read as ESM-2 results.",
            ),
            details={"representation": self.backend.fingerprint()},
        )

    def is_available(self) -> tuple[bool, str]:
        if self.backend.is_language_model:
            from ..features.esm import esm_available

            if not esm_available():
                return False, (
                    "torch and transformers are not installed, so ESM-2 embeddings cannot be"
                    " computed. Install with 'pip install \"assaypilot[esm]\"', or use the"
                    " physicochemical representation."
                )
        return True, ""

    def clone(self) -> EmbeddingRidgeMethod:
        return EmbeddingRidgeMethod(
            self.backend,
            cache=self.cache,
            alphas=self.alphas,
            n_components=self.n_components,
            batch_size=self.batch_size,
        )

    def _embed(self, sequences: Sequence[str]) -> np.ndarray:
        return embed_sequences(
            self.backend, sequences, cache=self.cache, batch_size=self.batch_size
        )

    def fit(
        self, sequences: Sequence[str], y: np.ndarray, *, groups: np.ndarray | None = None
    ) -> EmbeddingRidgeMethod:
        sequences = list(sequences)
        X = self._embed(sequences)
        y = np.asarray(y, dtype=float)

        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("reduce", "passthrough"),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )

        cv = _inner_cv(len(sequences), groups)

        # PCA must fit inside the *smallest inner training fold*, not the whole
        # training set: a component count that is fine for 90 rows can be
        # impossible for the 58 rows of a fold.
        if cv is None:
            smallest_fold = X.shape[0]
        else:
            smallest_fold = min(len(train_index) for train_index, _ in cv.split(X, y, groups))
        max_components = max(1, min(smallest_fold - 1, X.shape[1]))
        component_options: list[Any] = []
        for option in self.n_components:
            if option is None:
                component_options.append("passthrough")
            elif option <= max_components:
                component_options.append(PCA(n_components=option, random_state=0))
        if not component_options:
            component_options = ["passthrough"]

        if cv is None:
            alpha = self.alphas[len(self.alphas) // 2]
            pipeline.set_params(ridge__alpha=alpha, reduce=component_options[0])
            pipeline.fit(X, y)
            self._chosen = {
                "alpha": alpha,
                "selection": "defaults (too little data for inner cross-validation)",
            }
        else:
            search = GridSearchCV(
                pipeline,
                {"ridge__alpha": list(self.alphas), "reduce": component_options},
                cv=cv,
                scoring="neg_mean_squared_error",
                error_score="raise",
            )
            search.fit(X, y, groups=groups)
            pipeline = search.best_estimator_
            reduce_step = search.best_params_["reduce"]
            self._chosen = {
                "alpha": float(search.best_params_["ridge__alpha"]),
                "n_components": (
                    None if reduce_step == "passthrough" else int(reduce_step.n_components)
                ),
                "selection": f"grouped inner cross-validation ({cv.get_n_splits()} folds)",
            }

        self._pipeline = pipeline
        self._chosen["n_embedding_features"] = int(X.shape[1])
        return self

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("EmbeddingRidgeMethod.predict called before fit")
        X = self._embed(list(sequences))
        return np.asarray(self._pipeline.predict(X), dtype=float)

    def fit_details(self) -> dict[str, Any]:
        details = dict(self._chosen)
        details["representation"] = self.backend.fingerprint()
        return details

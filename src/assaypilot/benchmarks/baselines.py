"""External methods, reimplemented for comparison.

Nothing here is part of AssayPilot's product suite. The comparison suite that
runs for a real campaign is deliberately small and fixed; these exist only so
the benchmark can answer the question the suite cannot ask of itself — *does
AssayPilot do better than the tool a researcher would otherwise reach for?*

Each method states what it is a reimplementation of, what was matched, what
was not, and how a bad result should be read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..features.base import EmbeddingBackend, EmbeddingCache, embed_sequences
from ..models.base import Method, MethodInfo

#: Where the published defaults below were read from, and when. Pin this in any
#: report: a reimplementation is only checkable against a stated source.
EVOLVEPRO_SOURCE = (
    "github.com/mat10d/EvolvePro, evolvepro/src/model.py (top_layer), read 19 October 2026;"
    " Jiang et al., Science 2024, doi:10.1126/science.adr6006"
)


class EvolveProStyleMethod(Method):
    """A reimplementation of EVOLVEpro's published selection procedure.

    What EVOLVEpro does, from its source: embed variants with a protein
    language model, fit a random forest on the measured variants with no
    feature scaling and no dimensionality reduction, sort the remaining pool
    by predicted activity, and take the top of that ranking as the next round.
    The defaults reproduced here are its own —
    ``RandomForestRegressor(n_estimators=100, criterion="friedman_mse",
    max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0,
    bootstrap=True, random_state=1)`` — fitted on raw labels.

    **What is matched:** the regressor and its hyperparameters, the absence of
    preprocessing, and greedy top-k selection.

    **What is not matched, and why:**

    * *The representation.* EVOLVEpro supports several protein language
      models; this uses whichever backend the benchmark is already using for
      AssayPilot's own methods. That is deliberate — holding the embeddings
      fixed makes the comparison about the regression head and the selection
      rule rather than about which checkpoint someone downloaded.
    * *The first round.* EVOLVEpro can seed its first round with k-medoid
      "diverse medoids" over PCA'd embeddings. The replay harness gives every
      strategy the *same* randomly drawn initial set, because that pairing is
      what makes the comparison meaningful across seeds. EVOLVEpro's
      initialisation is therefore not exercised, and any advantage it confers
      is not represented here.

    **How to read a poor result:** as evidence about this reimplementation
    under these conditions, not about EVOLVEpro. It is their procedure as
    described, not their code, on a representation they did not choose, with
    an initialisation they did not pick.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        cache: EmbeddingCache | None = None,
        n_estimators: int = 100,
        random_state: int = 1,
        batch_size: int = 16,
        n_jobs: int | None = -1,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self._model: RandomForestRegressor | None = None

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key="evolvepro_style_rf",
            name="EVOLVEpro-style (random forest on embeddings)",
            kind="external_reimplementation",
            description=(
                "Random forest on frozen protein-language-model embeddings with no"
                " preprocessing, selecting greedily by predicted activity — EVOLVEpro's"
                " published procedure, reimplemented."
            ),
            uses_labels=True,
            complexity=3,
            caveats=(
                "A reimplementation of a published procedure, not the authors' code: a poor"
                " result is evidence about this implementation, not about EVOLVEpro.",
                "Runs on the same embeddings as the other methods in the comparison, which is"
                " not necessarily the protein language model EVOLVEpro would have used.",
                "EVOLVEpro's diverse-medoid initialisation is not used; every strategy in the"
                " replay shares one randomly drawn initial set so the seeds stay paired.",
            ),
            details={
                "source": EVOLVEPRO_SOURCE,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "preprocessing": "none, as published",
                "criterion": "squared_error (what EVOLVEpro's friedman_mse maps to)",
                "representation": self.backend.fingerprint(),
            },
        )

    def is_available(self) -> tuple[bool, str]:
        if self.backend.is_language_model:
            from ..features.esm import esm_available

            if not esm_available():
                return False, (
                    "torch and transformers are not installed, so the embeddings this method"
                    " needs cannot be computed"
                )
        return True, ""

    def clone(self) -> EvolveProStyleMethod:
        return EvolveProStyleMethod(
            self.backend,
            cache=self.cache,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_jobs=self.n_jobs,
        )

    def _embed(self, sequences: Sequence[str]) -> np.ndarray:
        return embed_sequences(
            self.backend, sequences, cache=self.cache, batch_size=self.batch_size
        )

    def fit(
        self, sequences: Sequence[str], y: np.ndarray, *, groups: np.ndarray | None = None
    ) -> EvolveProStyleMethod:
        del groups  # the published procedure fits on the rows it is given
        features = self._embed(list(sequences))
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            # EVOLVEpro passes criterion="friedman_mse". scikit-learn now
            # deprecates that name, stating the two were always equivalent for
            # regression forests, so this is the same split criterion under the
            # name that survives.
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(features, np.asarray(y, dtype=float))
        return self

    def predict(self, sequences: Sequence[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EvolveProStyleMethod.predict called before fit")
        return np.asarray(self._model.predict(self._embed(list(sequences))), dtype=float)

    def fit_details(self) -> dict[str, Any]:
        return {
            "source": EVOLVEPRO_SOURCE,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "representation": self.backend.fingerprint(),
        }

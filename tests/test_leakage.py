"""Leakage: the failures that quietly inflate every number downstream.

Four kinds are checked here:

1. **Replicate leakage** — a sequence (and all its replicates) appearing on
   both sides of a split.
2. **Fit-time leakage** — a method seeing held-out examples during fitting.
3. **Preprocessing leakage** — scaling or dimensionality reduction fitted over
   the whole dataset instead of inside the training fold.
4. **Vocabulary leakage** — the baseline's one-hot vocabulary built from
   substitutions it should not know about.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from assaypilot.evaluation.compare import evaluate_methods
from assaypilot.evaluation.splits import choose_split_plan
from assaypilot.features.base import embed_sequences
from assaypilot.features.onehot import SubstitutionOneHotEncoder
from assaypilot.features.physchem import PhysicochemicalBackend
from assaypilot.models.assay_trained import EmbeddingRidgeMethod
from assaypilot.models.base import Method, MethodInfo
from assaypilot.models.baseline import OneHotRidgeBaseline


class RecordingMethod(Method):
    """A method that records exactly which sequences it was fitted on."""

    def __init__(self) -> None:
        self.fitted_on: list[list[str]] = []
        self.predicted_on: list[list[str]] = []

    @property
    def info(self) -> MethodInfo:
        return MethodInfo(
            key="recording",
            name="Recording method",
            kind="baseline",
            description="records its inputs",
            uses_labels=True,
            complexity=1,
        )

    def clone(self) -> RecordingMethod:
        clone = RecordingMethod()
        clone.fitted_on = self.fitted_on
        clone.predicted_on = self.predicted_on
        return clone

    def fit(self, sequences, y, *, groups=None):  # noqa: ANN001
        self.fitted_on.append(list(sequences))
        self._mean = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, sequences):  # noqa: ANN001
        self.predicted_on.append(list(sequences))
        return np.full(len(list(sequences)), self._mean, dtype=float)


def test_no_sequence_appears_in_both_partitions(simulated_inputs):
    """Replicate leakage: the invariant every split builder must satisfy."""
    _, measurements, candidates = simulated_inputs
    frame = measurements.frame
    for policy in ("auto", "random"):
        plan = choose_split_plan(measurements, candidates, policy=policy)
        for split in plan.splits:
            train = set(frame.iloc[split.train_index]["seq_hash"])
            test = set(frame.iloc[split.test_index]["seq_hash"])
            assert not (train & test), f"{policy}/{split.name} leaks {len(train & test)} sequences"


def test_methods_are_fitted_only_on_training_rows(simulated_inputs):
    """Fit-time leakage: a method must never see a held-out sequence while fitting."""
    campaign, measurements, candidates = simulated_inputs
    plan = choose_split_plan(measurements, candidates)
    method = RecordingMethod()

    evaluate_methods([method], measurements, plan, campaign=campaign, k=4, n_bootstrap=10, seed=0)

    frame = measurements.frame
    assert len(method.fitted_on) == len(plan.splits)
    for split, fitted, predicted in zip(
        plan.splits, method.fitted_on, method.predicted_on, strict=True
    ):
        expected_train = set(frame.iloc[split.train_index]["sequence"])
        expected_test = set(frame.iloc[split.test_index]["sequence"])
        assert set(fitted) == expected_train
        assert set(predicted) == expected_test
        assert not set(fitted) & expected_test


def test_scaler_is_fitted_inside_the_training_fold(simulated_inputs, physchem_suite):
    """Preprocessing leakage: the scaler must know only the training rows."""
    _, measurements, _ = simulated_inputs
    frame = measurements.frame
    train = frame.iloc[: len(frame) // 2]
    held_out = frame.iloc[len(frame) // 2 :]

    backend = PhysicochemicalBackend(n_bins=4)
    method = EmbeddingRidgeMethod(backend, cache=None, n_components=(None,))
    method.fit(
        train["sequence"].tolist(),
        train["value"].to_numpy(dtype=float),
        groups=train["seq_hash"].to_numpy(),
    )

    fitted_scaler: StandardScaler = method._pipeline.named_steps["scale"]
    train_embeddings = embed_sequences(backend, train["sequence"].tolist())
    all_embeddings = embed_sequences(backend, frame["sequence"].tolist())

    np.testing.assert_allclose(fitted_scaler.mean_, train_embeddings.mean(axis=0), rtol=1e-5)
    # And it is genuinely different from what a leaky, dataset-wide fit would give.
    assert not np.allclose(fitted_scaler.mean_, all_embeddings.mean(axis=0), rtol=1e-6)
    assert len(held_out) > 0


def test_onehot_vocabulary_excludes_unseen_substitutions(campaign, reference, mutator):
    """Vocabulary leakage: substitutions only present in held-out data stay unknown."""
    train_sequences = [mutator(reference, (3, "G")), mutator(reference, (5, "W"))]
    test_sequence = mutator(reference, (7, "D"))

    encoder = SubstitutionOneHotEncoder(reference)
    encoder.fit(train_sequences)

    assert (7, "D") not in encoder.vocabulary_
    assert encoder.unseen_substitution_fraction([test_sequence]) == 1.0

    encoded = encoder.transform([test_sequence])
    # Only the substitution-count feature is non-zero for a fully unseen variant.
    assert encoded[0, :-1].sum() == 0
    assert encoded[0, -1] == 1


def test_baseline_reports_unseen_substitutions_after_fit(campaign, reference, mutator):
    train = [
        mutator(reference, (3, "G")),
        mutator(reference, (3, "A")),
        mutator(reference, (5, "W")),
    ]
    baseline = OneHotRidgeBaseline(reference, alphas=(1.0,))
    baseline.fit(train, np.array([1.0, 0.5, 2.0]))
    assert baseline.unseen_substitution_fraction([mutator(reference, (9, "K"))]) == 1.0
    assert baseline.unseen_substitution_fraction([mutator(reference, (3, "G"))]) == 0.0


def test_predict_before_fit_is_an_error(reference):
    with pytest.raises(RuntimeError, match="before fit"):
        OneHotRidgeBaseline(reference).predict([reference])
    with pytest.raises(RuntimeError, match="before fit"):
        EmbeddingRidgeMethod(PhysicochemicalBackend(n_bins=2)).predict([reference])

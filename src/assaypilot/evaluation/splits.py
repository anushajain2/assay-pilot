"""Evaluation splits that follow the step the researcher is about to take.

The default evidence for "will this method help me choose the next batch?" is
*not* a random split. A random split asks whether a model interpolates among
variants like the ones it already saw. The next batch is usually further away
than that.

So the preference order is:

1. **Round split** — train on earlier rounds, evaluate on a later round. This
   is the closest analogue of the real task and is used whenever usable round
   labels exist.
2. **Mutation-distance split** — hold out the variants that are as far from the
   reference as the candidate pool is. Used when there are no rounds but the
   candidates are more mutated than most measured variants.
3. **Mutation-pattern split** — hold out whole sets of positions, so the
   evaluation measures prediction at positions never seen in training.
4. **Random split (grouped)** — an interpolation diagnostic. Always labelled as
   such, never presented as evidence about distant candidates.

Two invariants hold for every split produced here:

* identical sequences and all their replicates stay in the same partition
  (grouping is by ``seq_hash``);
* no example appears in both sides of a split.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.loaders import CandidateSet, MeasurementSet, _round_sort_key
from ..errors import InsufficientDataError

SplitKind = str  # "round" | "mutation_distance" | "mutation_pattern" | "random"

#: Minimum partition sizes below which a split is not produced at all.
MIN_TRAIN_SEQUENCES = 8
MIN_TEST_SEQUENCES = 5


@dataclass(frozen=True)
class Split:
    """One train/test partition over row indices of a measurement table."""

    name: str
    kind: SplitKind
    train_index: np.ndarray
    test_index: np.ndarray
    description: str
    extrapolation: bool
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(len(self.train_index))

    @property
    def n_test(self) -> int:
        return int(len(self.test_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "extrapolation": self.extrapolation,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_index": [int(i) for i in self.train_index],
            "test_index": [int(i) for i in self.test_index],
            "details": dict(self.details),
        }


@dataclass
class SplitPlan:
    """The chosen evaluation protocol, with the reasoning that produced it."""

    kind: SplitKind
    splits: list[Split]
    rationale: str
    extrapolation: bool
    warnings: list[str] = field(default_factory=list)
    considered: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.splits)

    @property
    def is_interpolation_only(self) -> bool:
        """True when the only evidence available is a random (interpolation) split."""
        return self.kind == "random"

    @property
    def label(self) -> str:
        """Short label printed next to every number produced under this plan."""
        return "extrapolation evaluation" if self.extrapolation else "interpolation diagnostic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "rationale": self.rationale,
            "extrapolation": self.extrapolation,
            "n_splits": len(self.splits),
            "warnings": list(self.warnings),
            "considered": dict(self.considered),
            "splits": [s.to_dict() for s in self.splits],
        }


def _check_no_group_overlap(frame: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> None:
    """Guard against the leakage this module exists to prevent."""
    train_groups = set(frame.iloc[train]["seq_hash"])
    test_groups = set(frame.iloc[test]["seq_hash"])
    overlap = train_groups & test_groups
    if overlap:  # pragma: no cover - a bug in a split builder, not a user error
        raise AssertionError(f"split leakage: {len(overlap)} sequence(s) appear in both partitions")


def _group_sizes(frame: pd.DataFrame, index: np.ndarray) -> int:
    return int(frame.iloc[index]["seq_hash"].nunique()) if len(index) else 0


def round_splits(measurements: MeasurementSet) -> list[Split]:
    """Forward-chaining splits over round labels: earlier rounds predict the next one."""
    frame = measurements.frame
    if "round_id" not in frame or frame["round_id"].isna().all():
        return []

    rounds = sorted(frame["round_id"].dropna().unique().tolist(), key=_round_sort_key)
    if len(rounds) < 2:
        return []

    splits: list[Split] = []
    for i in range(1, len(rounds)):
        train_rounds = rounds[:i]
        test_round = rounds[i]
        train_index = np.flatnonzero(frame["round_id"].isin(train_rounds).to_numpy())
        test_index = np.flatnonzero((frame["round_id"] == test_round).to_numpy())

        # A sequence remeasured in a later round would otherwise sit on both
        # sides of the split. It stays with the earlier rounds the model trains
        # on, and is removed from the evaluation set rather than scored as if
        # it were new.
        trained_hashes = set(frame.iloc[train_index]["seq_hash"])
        test_hashes = frame.iloc[test_index]["seq_hash"].to_numpy()
        keep = np.array([h not in trained_hashes for h in test_hashes], dtype=bool)
        n_remeasured = int((~keep).sum())
        test_index = test_index[keep]

        if (
            _group_sizes(frame, train_index) < MIN_TRAIN_SEQUENCES
            or _group_sizes(frame, test_index) < MIN_TEST_SEQUENCES
        ):
            continue
        _check_no_group_overlap(frame, train_index, test_index)
        listed = ", ".join(str(r) for r in train_rounds)
        splits.append(
            Split(
                name=f"round_{test_round}",
                kind="round",
                train_index=train_index,
                test_index=test_index,
                description=(
                    f"trained on round(s) {listed}, evaluated on round {test_round}"
                    + (
                        f" (excluding {n_remeasured} row(s) whose sequence was already"
                        " measured in an earlier round)"
                        if n_remeasured
                        else ""
                    )
                ),
                extrapolation=True,
                details={
                    "train_rounds": [str(r) for r in train_rounds],
                    "test_round": str(test_round),
                    "n_remeasured_excluded": n_remeasured,
                },
            )
        )
    return splits


def mutation_distance_split(
    measurements: MeasurementSet,
    *,
    target_distance: int | None = None,
    min_test_fraction: float = 0.15,
    max_test_fraction: float = 0.5,
) -> list[Split]:
    """Hold out the most-substituted variants, matching the candidate pool's distance.

    ``target_distance`` is normally the median number of substitutions in the
    candidate pool: the evaluation then answers "how well does this method
    predict variants as mutated as the ones I am about to order?".
    """
    frame = measurements.frame
    if frame.empty:
        return []
    n_subs = frame["n_substitutions"].to_numpy()
    distinct = sorted(set(int(v) for v in n_subs))
    if len(distinct) < 2:
        return []

    n_groups_total = frame["seq_hash"].nunique()
    best: tuple[int, np.ndarray, np.ndarray] | None = None

    thresholds = distinct[1:]
    if target_distance is not None:
        # Prefer thresholds at or below the candidate distance, closest first.
        thresholds = sorted(thresholds, key=lambda t: (abs(t - target_distance), -t))

    for threshold in thresholds:
        test_index = np.flatnonzero(n_subs >= threshold)
        train_index = np.flatnonzero(n_subs < threshold)
        n_test_groups = _group_sizes(frame, test_index)
        n_train_groups = _group_sizes(frame, train_index)
        if n_train_groups < MIN_TRAIN_SEQUENCES or n_test_groups < MIN_TEST_SEQUENCES:
            continue
        fraction = n_test_groups / n_groups_total
        if not (min_test_fraction <= fraction <= max_test_fraction):
            continue
        best = (threshold, train_index, test_index)
        break

    if best is None:
        return []

    threshold, train_index, test_index = best
    _check_no_group_overlap(frame, train_index, test_index)
    target_note = (
        f" The candidate pool has a median of {target_distance} substitutions."
        if target_distance is not None
        else ""
    )
    return [
        Split(
            name=f"distance_ge_{threshold}",
            kind="mutation_distance",
            train_index=train_index,
            test_index=test_index,
            description=(
                f"trained on variants with fewer than {threshold} substitutions,"
                f" evaluated on variants with {threshold} or more.{target_note}"
            ),
            extrapolation=True,
            details={"threshold": int(threshold), "target_distance": target_distance},
        )
    ]


def mutation_pattern_splits(
    measurements: MeasurementSet,
    *,
    n_splits: int = 3,
    seed: int = 0,
) -> list[Split]:
    """Hold out whole groups of mutated positions.

    For each fold a set of positions is held out. Test variants are those that
    mutate at least one held-out position; training variants are those that
    mutate none of them. Variants that fall in neither partition (they mix
    held-out and training positions) are excluded from that fold and counted in
    ``details['n_unassigned']``, because keeping them would leak the held-out
    positions into training.
    """
    frame = measurements.frame
    if frame.empty:
        return []

    position_sets = [
        frozenset(int(p[1:-1]) for p in str(m).split(":") if m != "WT" and p)
        for m in frame["mutations"]
    ]
    all_positions = sorted({p for s in position_sets for p in s})
    if len(all_positions) < n_splits * 2:
        return []

    rng = np.random.default_rng(seed)
    shuffled = list(all_positions)
    rng.shuffle(shuffled)
    folds = [set(shuffled[i::n_splits]) for i in range(n_splits)]

    splits: list[Split] = []
    for i, held_out in enumerate(folds, start=1):
        test_mask = np.array([bool(s & held_out) for s in position_sets])
        train_mask = np.array([bool(s) and not (s & held_out) for s in position_sets])
        test_index = np.flatnonzero(test_mask)
        train_index = np.flatnonzero(train_mask)
        if (
            _group_sizes(frame, train_index) < MIN_TRAIN_SEQUENCES
            or _group_sizes(frame, test_index) < MIN_TEST_SEQUENCES
        ):
            continue
        _check_no_group_overlap(frame, train_index, test_index)
        n_unassigned = int(len(frame) - len(test_index) - len(train_index))
        splits.append(
            Split(
                name=f"positions_fold{i}",
                kind="mutation_pattern",
                train_index=train_index,
                test_index=test_index,
                description=(
                    f"trained on variants that leave {len(held_out)} held-out positions"
                    " untouched, evaluated on variants that mutate at least one of them"
                ),
                extrapolation=True,
                details={
                    "n_held_out_positions": len(held_out),
                    "n_unassigned": n_unassigned,
                },
            )
        )
    return splits


def random_group_splits(
    measurements: MeasurementSet, *, n_splits: int = 5, seed: int = 0
) -> list[Split]:
    """Grouped K-fold over sequences: the interpolation diagnostic."""
    frame = measurements.frame
    groups = frame["seq_hash"].to_numpy()
    unique = np.unique(groups)
    n_splits = int(min(n_splits, len(unique)))
    if n_splits < 2 or len(unique) < MIN_TRAIN_SEQUENCES + MIN_TEST_SEQUENCES:
        return []

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    assignment = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_of_row = np.array([assignment[g] for g in groups])

    splits: list[Split] = []
    for fold in range(n_splits):
        test_index = np.flatnonzero(fold_of_row == fold)
        train_index = np.flatnonzero(fold_of_row != fold)
        if (
            _group_sizes(frame, train_index) < MIN_TRAIN_SEQUENCES
            or _group_sizes(frame, test_index) < 2
        ):
            continue
        _check_no_group_overlap(frame, train_index, test_index)
        splits.append(
            Split(
                name=f"random_fold{fold + 1}",
                kind="random",
                train_index=train_index,
                test_index=test_index,
                description=(
                    f"random grouped fold {fold + 1} of {n_splits}; replicates and identical"
                    " sequences kept together"
                ),
                extrapolation=False,
                details={"fold": fold + 1, "n_folds": n_splits, "seed": seed},
            )
        )
    return splits


def candidate_target_distance(candidates: CandidateSet | None) -> int | None:
    """Median substitution count of the candidate pool, used to aim the distance split."""
    if candidates is None or candidates.frame.empty:
        return None
    return int(np.median(candidates.frame["n_substitutions"].to_numpy()))


def choose_split_plan(
    measurements: MeasurementSet,
    candidates: CandidateSet | None = None,
    *,
    policy: str = "auto",
    seed: int = 0,
) -> SplitPlan:
    """Pick the evaluation protocol that best matches the next experimental step.

    Parameters
    ----------
    policy:
        ``"auto"`` (default) applies the preference order described in the
        module docstring. ``"round"``, ``"mutation_distance"``,
        ``"mutation_pattern"`` and ``"random"`` force one protocol and raise
        :class:`~assaypilot.errors.InsufficientDataError` if it cannot be built.
    """
    considered: dict[str, str] = {}
    warnings: list[str] = []
    target = candidate_target_distance(candidates)

    builders = {
        "round": lambda: round_splits(measurements),
        "mutation_distance": lambda: mutation_distance_split(measurements, target_distance=target),
        "mutation_pattern": lambda: mutation_pattern_splits(measurements, seed=seed),
        "random": lambda: random_group_splits(measurements, seed=seed),
    }

    if policy != "auto":
        if policy not in builders:
            raise ValueError(f"unknown split policy {policy!r}")
        splits = builders[policy]()
        if not splits:
            raise InsufficientDataError(
                f"the {policy!r} split could not be built from these measurements"
                f" (need at least {MIN_TRAIN_SEQUENCES} training and {MIN_TEST_SEQUENCES}"
                " evaluation sequences, and the labels the split depends on)"
            )
        return SplitPlan(
            kind=policy,
            splits=splits,
            rationale=f"split protocol forced to {policy!r} by the caller",
            extrapolation=splits[0].extrapolation,
            warnings=warnings,
            considered=considered,
        )

    rounds = round_splits(measurements)
    if rounds:
        return SplitPlan(
            kind="round",
            splits=rounds,
            rationale=(
                "Usable round labels are present, so the evaluation reproduces the real task:"
                " earlier rounds are used to predict a later round."
            ),
            extrapolation=True,
            warnings=warnings,
            considered=considered,
        )
    considered["round"] = (
        "not used: fewer than two rounds with enough measurements, or no round labels"
    )

    distance = mutation_distance_split(measurements, target_distance=target)
    if distance:
        rationale = (
            "No usable round labels, so the evaluation holds out the most-substituted"
            " measured variants"
        )
        if target is not None:
            rationale += f", matching the candidate pool's median of {target} substitutions"
        return SplitPlan(
            kind="mutation_distance",
            splits=distance,
            rationale=rationale + ".",
            extrapolation=True,
            warnings=warnings,
            considered=considered,
        )
    considered["mutation_distance"] = (
        "not used: substitution counts do not allow a holdout of a usable size"
    )

    pattern = mutation_pattern_splits(measurements, seed=seed)
    if pattern:
        return SplitPlan(
            kind="mutation_pattern",
            splits=pattern,
            rationale=(
                "No round labels and no usable distance holdout, so whole sets of mutated"
                " positions are held out; the evaluation measures prediction at positions"
                " not seen in training."
            ),
            extrapolation=True,
            warnings=warnings,
            considered=considered,
        )
    considered["mutation_pattern"] = "not used: too few distinct mutated positions"

    random_folds = random_group_splits(measurements, seed=seed)
    if random_folds:
        warnings.append(
            "Only a random split was possible. These numbers describe interpolation among"
            " variants like the measured ones. They are not evidence about candidates that"
            " are further away, and they are labelled as an interpolation diagnostic"
            " throughout the report."
        )
        return SplitPlan(
            kind="random",
            splits=random_folds,
            rationale=(
                "No round, distance, or position holdout could be built, so only a random"
                " grouped split is available."
            ),
            extrapolation=False,
            warnings=warnings,
            considered=considered,
        )
    considered["random"] = "not used: too few distinct sequences"

    raise InsufficientDataError(
        "No evaluation split could be built from these measurements. At least"
        f" {MIN_TRAIN_SEQUENCES + MIN_TEST_SEQUENCES} distinct measured sequences are needed"
        " before any method comparison is meaningful. You can still prepare data and produce"
        " an exploratory pretrained ranking, which AssayPilot labels as unvalidated."
    )


def describe_splits(splits: Iterable[Split]) -> pd.DataFrame:
    """Tabular summary of a split plan, for the review screen."""
    rows = [
        {
            "split": s.name,
            "kind": s.kind,
            "n_train": s.n_train,
            "n_test": s.n_test,
            "extrapolation": s.extrapolation,
            "description": s.description,
        }
        for s in splits
    ]
    return pd.DataFrame(rows)


def training_sequences(measurements: MeasurementSet, splits: Sequence[Split]) -> list[str]:
    """Sequences used for training across a split plan (for coverage checks)."""
    frame = measurements.frame
    index = sorted({int(i) for s in splits for i in s.train_index})
    return frame.iloc[index]["sequence"].tolist()

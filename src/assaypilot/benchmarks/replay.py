"""Retrospective replay: evaluate the whole decision process, not just a model.

A replay takes a fully measured public dataset, hides it, and plays the
campaign forward: reveal a small initial set, fit, choose a fixed batch,
reveal only that batch's outcomes, repeat. Every strategy gets the same
candidate pool, the same initial observations, the same budget, and the same
seeds, so the comparison is about the decision rule rather than the luck of
the draw.

Two limits are structural and are printed with every result:

* a replay can only "select" variants whose outcome the dataset already
  contains, so the candidate pool is not the pool a real campaign would face;
* the exercise is retrospective — it shows what the strategy *would* have
  chosen, not what a new experiment would have produced.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..campaign import Campaign
from ..models.base import Method
from ..models.registry import SuiteConfig, build_cache, build_methods, split_by_availability
from ..selection.policies import DiversityPolicy, SelectionPolicy, TopRankedPolicy

#: Printed beside every replay table.
REPLAY_CAVEAT = (
    "Retrospective replay on an already-measured dataset. Selection is restricted to variants"
    " whose outcome the dataset contains, and no new experiment was performed."
)


@dataclass
class ReplayConfig:
    """Settings shared by every strategy in a replay."""

    n_initial: int = 48
    batch_size: int = 24
    n_rounds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    diversity_min_distance: int | None = 2
    suite: SuiteConfig = field(default_factory=lambda: SuiteConfig(representation="physchem"))
    include_random: bool = True
    include_diversity_variant: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = {
            "n_initial": self.n_initial,
            "batch_size": self.batch_size,
            "n_rounds": self.n_rounds,
            "seeds": list(self.seeds),
            "diversity_min_distance": self.diversity_min_distance,
            "include_random": self.include_random,
            "include_diversity_variant": self.include_diversity_variant,
            "suite": self.suite.to_dict(),
        }
        return data


@dataclass
class ReplayResult:
    """Per-round outcomes for every strategy and seed."""

    frame: pd.DataFrame
    config: ReplayConfig
    campaign_id: str
    caveat: str = REPLAY_CAVEAT

    def summary(self) -> pd.DataFrame:
        """Mean and spread across seeds at the final round of each strategy."""
        if self.frame.empty:
            return self.frame
        final = self.frame.sort_values("round").groupby(["strategy", "seed"]).tail(1)
        grouped = final.groupby("strategy").agg(
            best_selected_mean=("best_selected", "mean"),
            best_selected_sd=("best_selected", "std"),
            best_found_mean=("best_found", "mean"),
            best_found_sd=("best_found", "std"),
            hits_selected_mean=("hits_selected", "mean"),
            n_hits_mean=("cumulative_hits", "mean"),
            experiments=("cumulative_experiments", "mean"),
            seconds_mean=("seconds", "mean"),
        )
        return grouped.sort_values("best_selected_mean", ascending=False).reset_index()

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "config": self.config.to_dict(),
            "caveat": self.caveat,
            "summary": self.summary().to_dict(orient="records"),
        }


def _unique_pool(measurements: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence: replicates averaged, so a replay never re-measures noise."""
    pool = (
        measurements.groupby("sequence", as_index=False)
        .agg(value=("value", "mean"), variant_id=("variant_id", "first"))
        .reset_index(drop=True)
    )
    return pool


def _random_strategy(rng: np.random.Generator) -> Callable[[pd.DataFrame, int], list[int]]:
    def choose(remaining: pd.DataFrame, batch_size: int) -> list[int]:
        size = min(batch_size, len(remaining))
        return list(rng.choice(len(remaining), size=size, replace=False))

    return choose


def _method_strategy(
    method: Method,
    *,
    sign: int,
    policy: SelectionPolicy,
) -> Callable[[pd.DataFrame, pd.DataFrame, int], list[int]]:
    def choose(observed: pd.DataFrame, remaining: pd.DataFrame, batch_size: int) -> list[int]:
        fitted = method.clone()
        fitted.fit(
            observed["sequence"].tolist(),
            sign * observed["value"].to_numpy(dtype=float),
            groups=observed["sequence"].to_numpy(),
        )
        scores = np.asarray(fitted.predict(remaining["sequence"].tolist()), dtype=float)
        ranked = remaining.copy()
        ranked["score"] = scores
        ranked["candidate_id"] = ranked.index.astype(str)
        ranked = ranked.sort_values(["score", "candidate_id"], ascending=[False, True])
        outcome = policy.select(ranked.reset_index(drop=True), batch_size)
        return [int(i) for i in outcome.selected["candidate_id"].astype(int)]

    return choose


def replay_campaign(
    campaign: Campaign,
    measurements: pd.DataFrame,
    config: ReplayConfig | None = None,
) -> ReplayResult:
    """Play a measured dataset forward, strategy by strategy.

    ``measurements`` must contain ``sequence`` and ``value`` columns covering
    the whole pool: those values are the oracle the replay reveals one batch
    at a time.
    """
    config = config or ReplayConfig()
    pool = _unique_pool(measurements)
    if len(pool) < config.n_initial + config.batch_size:
        raise ValueError(
            f"the pool has {len(pool)} distinct sequences, too few for"
            f" {config.n_initial} initial observations plus a batch of {config.batch_size}"
        )

    sign = campaign.sign
    threshold = campaign.success_threshold
    cache = build_cache(config.suite)
    methods, unavailable = split_by_availability(build_methods(campaign, config.suite, cache=cache))

    strategies: dict[str, dict[str, Any]] = {}
    if config.include_random:
        strategies["random"] = {"kind": "random"}
    for method in methods:
        strategies[method.info.key] = {"kind": "method", "method": method, "policy": "top_ranked"}
    if config.include_diversity_variant and config.diversity_min_distance:
        trained = [m for m in methods if m.info.kind == "assay_trained"]
        for method in trained:
            strategies[f"{method.info.key}+diversity"] = {
                "kind": "method",
                "method": method,
                "policy": "diversity",
            }

    rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        rng_initial = np.random.default_rng(seed)
        initial_index = rng_initial.choice(len(pool), size=config.n_initial, replace=False)
        initial = set(int(i) for i in initial_index)

        for name, spec in strategies.items():
            rng = np.random.default_rng(seed + 1000)
            observed_index = set(initial)
            policy: SelectionPolicy = (
                DiversityPolicy(config.diversity_min_distance)
                if spec.get("policy") == "diversity" and config.diversity_min_distance
                else TopRankedPolicy()
            )
            selected_values: list[float] = []
            random_choose = _random_strategy(rng)
            method_choose = (
                _method_strategy(spec["method"], sign=sign, policy=policy)
                if spec["kind"] == "method"
                else None
            )

            for round_number in range(1, config.n_rounds + 1):
                observed = pool.iloc[sorted(observed_index)]
                remaining = pool.drop(index=sorted(observed_index)).reset_index()
                if remaining.empty:
                    break

                start = time.perf_counter()
                if spec["kind"] == "random":
                    picks = random_choose(remaining, config.batch_size)
                else:
                    assert method_choose is not None
                    picks = method_choose(observed, remaining, config.batch_size)
                seconds = time.perf_counter() - start

                chosen_rows = remaining.iloc[picks]
                chosen_pool_index = [int(i) for i in chosen_rows["index"]]
                observed_index.update(chosen_pool_index)

                revealed = pool.iloc[sorted(observed_index)]
                batch_values = chosen_rows["value"].to_numpy(dtype=float)
                selected_values.extend(batch_values.tolist())
                best = max if campaign.objective == "maximize" else min
                best_found = float(best(revealed["value"]))
                # best_found includes the revealed initial set, so it saturates
                # whenever that set already contained the pool's best variant —
                # and then every strategy looks identical. best_selected counts
                # only what the strategy itself chose, which keeps the
                # comparison informative.
                best_selected = float(best(selected_values)) if selected_values else float("nan")
                hits = (
                    int(revealed["value"].map(campaign.is_hit).sum())
                    if threshold is not None
                    else 0
                )
                selected_hits = (
                    int(sum(campaign.is_hit(v) for v in selected_values))
                    if threshold is not None
                    else 0
                )

                rows.append(
                    {
                        "campaign_id": campaign.campaign_id,
                        "strategy": name,
                        "seed": seed,
                        "round": round_number,
                        "batch_size": len(chosen_pool_index),
                        "batch_best": float(batch_values.max()) if len(batch_values) else np.nan,
                        "batch_mean": float(batch_values.mean()) if len(batch_values) else np.nan,
                        "best_found": best_found,
                        "best_selected": best_selected,
                        "cumulative_hits": hits,
                        "hits_selected": selected_hits,
                        "cumulative_experiments": len(observed_index),
                        "seconds": seconds,
                    }
                )

    frame = pd.DataFrame(rows)
    result = ReplayResult(frame=frame, config=config, campaign_id=campaign.campaign_id)
    if unavailable:
        result.caveat += " Methods not available in this environment: " + "; ".join(
            f"{k} ({v})" for k, v in unavailable.items()
        )
    return result


def compare_strategies(
    result: ReplayResult, *, reference: str = "random", metric: str = "best_selected"
) -> pd.DataFrame:
    """Difference from a reference strategy at the final round, per seed.

    Paired by seed, because the seed fixes both the initial observations and
    the pool order: an unpaired comparison would mostly measure that.

    The default metric is ``best_selected`` — the best variant the strategy
    itself chose. ``best_found`` includes the shared initial set, so it can
    saturate and hide every difference between strategies.
    """
    frame = result.frame
    if frame.empty:
        return frame
    final = frame.sort_values("round").groupby(["strategy", "seed"]).tail(1)
    pivot = final.pivot(index="seed", columns="strategy", values=metric)
    if reference not in pivot.columns:
        raise ValueError(f"no strategy named {reference!r} in this replay")
    rows = []
    for strategy in pivot.columns:
        if strategy == reference:
            continue
        differences = (pivot[strategy] - pivot[reference]).to_numpy(dtype=float)
        rows.append(
            {
                "strategy": strategy,
                "reference": reference,
                "metric": metric,
                "mean_difference": float(np.nanmean(differences)),
                "sd_difference": float(np.nanstd(differences, ddof=1))
                if len(differences) > 1
                else float("nan"),
                "n_seeds": int(np.sum(np.isfinite(differences))),
                "wins": int(np.sum(differences > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_difference", ascending=False).reset_index(drop=True)


def strategy_names(config: ReplayConfig, campaign: Campaign) -> Sequence[str]:
    """Names of the strategies a replay would run with this configuration."""
    methods, _ = split_by_availability(build_methods(campaign, config.suite))
    names = ["random"] if config.include_random else []
    names += [m.info.key for m in methods]
    if config.include_diversity_variant and config.diversity_min_distance:
        names += [f"{m.info.key}+diversity" for m in methods if m.info.kind == "assay_trained"]
    return names

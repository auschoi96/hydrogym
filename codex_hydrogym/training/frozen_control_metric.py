"""Frozen, deterministic policy-quality metric for Kolmogorov control.

This module is outside the candidate-reward path and is hash-pinned by
``test/codex_hydrogym/test_frozen_control_metric_guard.py``. A reward-writing
agent must not edit this file or its guard. Either edit is an explicit,
reviewable protocol change rather than a reward experiment.

For seed ``s``, at a preregistered matched mean action-L1 cost ``C*``, the
control-quality score is normalized TKE reduction::

    q_s = (TKE_uncontrolled,s - TKE_candidate,s) / TKE_uncontrolled,s

Every candidate/seed must satisfy ``abs(C_s - C*) <= tolerance * C*``. The
comparison is therefore at matched control authority, not an exchange rate
chosen after seeing candidates. Higher ``q_s`` is better; zero means no TKE
reduction. HydroGym already computes trajectory-mean TKE in
``hydrogym/jax/envs/kolmogorov.py:562-567`` from the TKE implementation at
``hydrogym/jax/utils/utils.py:180-199``. It exposes that quantity and action
L1 at ``hydrogym/jax/envs/kolmogorov.py:646-649``. Training validation already
requires both fields at ``codex_hydrogym/training/validation.py:150-151``.

The score never consumes the candidate reward. It and its ranking are thus
invariant to every positive affine reward transformation ``R' = a R + b``,
``a > 0``. Pearson reward-vs-metric correlation is also invariant to that
transformation and is reported only as a hacking diagnostic. Pairwise flags
identify candidates whose mean candidate reward is higher while their frozen
metric is lower.

Limitations: the metric measures TKE suppression only at the frozen action-L1
operating point. It cannot compare unmatched control budgets, certify
stability outside measured rollouts, detect unmeasured actuator power or
state-constraint violations, establish causality, or generalize beyond the
sampled seeds. Action L1 is control authority, not physical actuator energy.
The uncontrolled denominator must be positive. Exactly four independent
seed clusters are required because the reused frozen 95% t critical is for
four clusters; phases/windows must be reduced within seed before calling this
API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import correlation, fmean
from typing import Sequence

from codex_hydrogym.gate0.ensemble_diagnostic import EnsembleDiagnosticSpec, _mean_ci

FROZEN_SEED_COUNT = 4
FROZEN_T_CRITICAL_95 = EnsembleDiagnosticSpec().seed_cluster_t_critical_95


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class SeedOutcome:
    """Seed-cluster aggregates from matched uncontrolled/candidate rollouts."""

    seed: int
    uncontrolled_mean_tke: float
    candidate_mean_tke: float
    mean_control_l1: float
    mean_candidate_reward: float

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for field_name in (
            "uncontrolled_mean_tke",
            "candidate_mean_tke",
            "mean_control_l1",
            "mean_candidate_reward",
        ):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        if self.uncontrolled_mean_tke <= 0.0:
            raise ValueError("uncontrolled_mean_tke must be positive")
        if self.candidate_mean_tke < 0.0:
            raise ValueError("candidate_mean_tke must be nonnegative")
        if self.mean_control_l1 < 0.0:
            raise ValueError("mean_control_l1 must be nonnegative")


@dataclass(frozen=True)
class CandidateOutcomes:
    """All independent seed clusters for one reward-trained policy."""

    candidate: str
    outcomes: tuple[SeedOutcome, ...]

    def __post_init__(self) -> None:
        if not self.candidate.strip():
            raise ValueError("candidate must be nonempty")
        if len(self.outcomes) != FROZEN_SEED_COUNT:
            raise ValueError(
                f"comparison requires exactly {FROZEN_SEED_COUNT} independent seeds (and refuses fewer than 3)"
            )
        seeds = [outcome.seed for outcome in self.outcomes]
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate outcomes require distinct seeds")


def compare_candidates(
    candidates: Sequence[CandidateOutcomes],
    *,
    matched_mean_control_l1: float,
    relative_cost_tolerance: float = 0.05,
) -> dict[str, object]:
    """Compare policies on frozen TKE reduction and expose reward divergence."""
    target_cost = _finite(matched_mean_control_l1, "matched_mean_control_l1")
    tolerance = _finite(relative_cost_tolerance, "relative_cost_tolerance")
    if target_cost <= 0.0:
        raise ValueError("matched_mean_control_l1 must be positive")
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative_cost_tolerance must be in [0, 1)")
    if len(candidates) < 2:
        raise ValueError("reward-vs-metric diagnostics require at least two candidates")
    names = [candidate.candidate for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be distinct")

    expected_seeds = {outcome.seed for outcome in candidates[0].outcomes}
    allowed_difference = tolerance * target_cost
    results: list[dict[str, object]] = []
    for candidate in candidates:
        if {outcome.seed for outcome in candidate.outcomes} != expected_seeds:
            raise ValueError("all candidates must use the identical seed clusters")
        ordered = sorted(candidate.outcomes, key=lambda outcome: outcome.seed)
        for outcome in ordered:
            if abs(outcome.mean_control_l1 - target_cost) > allowed_difference:
                raise ValueError(
                    f"{candidate.candidate} seed {outcome.seed} has unmatched mean_control_l1: "
                    f"{outcome.mean_control_l1} vs {target_cost}"
                )
        per_seed = [
            {
                "seed": outcome.seed,
                "normalized_tke_reduction": (outcome.uncontrolled_mean_tke - outcome.candidate_mean_tke)
                / outcome.uncontrolled_mean_tke,
                "mean_control_l1": outcome.mean_control_l1,
                "candidate_reward": outcome.mean_candidate_reward,
            }
            for outcome in ordered
        ]
        distribution = [float(item["normalized_tke_reduction"]) for item in per_seed]
        results.append(
            {
                "candidate": candidate.candidate,
                "per_seed": per_seed,
                "clustered_interval_95": _mean_ci(distribution, FROZEN_T_CRITICAL_95),
                "mean_candidate_reward": fmean(outcome.mean_candidate_reward for outcome in ordered),
            }
        )

    rewards = [float(result["mean_candidate_reward"]) for result in results]
    metrics = [float(result["clustered_interval_95"]["mean"]) for result in results]
    reward_metric_correlation = None
    if len(set(rewards)) > 1 and len(set(metrics)) > 1:
        reward_metric_correlation = correlation(rewards, metrics)

    flagged = sorted(
        str(result["candidate"])
        for result in results
        if any(
            float(result["mean_candidate_reward"]) > float(other["mean_candidate_reward"])
            and float(result["clustered_interval_95"]["mean"]) < float(other["clustered_interval_95"]["mean"])
            for other in results
        )
    )
    return {
        "metric": "normalized_tke_reduction_at_matched_action_l1",
        "matched_mean_control_l1": target_cost,
        "relative_cost_tolerance": tolerance,
        "seed_count": FROZEN_SEED_COUNT,
        "candidates": results,
        "reward_metric_correlation": reward_metric_correlation,
        "reward_metric_diverged": reward_metric_correlation is not None and reward_metric_correlation < 0.0,
        "reward_hacking_candidates": flagged,
    }

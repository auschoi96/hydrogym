"""Null and positive control runs over the project's paired-delta instrument.

The statistic under test is reused, not reimplemented and not mocked:

- per-group paired deltas follow the frozen MemAlign protocol expression
  ``deltas[group] = arm_b_score - arm_a_score`` and ``paired_mean_delta`` as
  computed in ``codex_hydrogym/notebooks/coding_agent_memalign_proof.py``
  (that notebook executes at import time on Databricks, so its pairing
  expression is reproduced verbatim here rather than imported);
- the group-clustered 95% interval is ``_mean_ci`` from
  ``codex_hydrogym.gate0.ensemble_diagnostic``, the existing clustered
  mean-interval statistic the project already uses for per-cluster effects;
- the 95% t critical for ten clusters (df=9) is the frozen
  ``seed_cluster_t_critical_95`` carried by
  ``codex_hydrogym.gate0.ensemble_replication.EnsembleReplicationSpec``;
- the decision rule is the protocol's: the interval must be wholly above
  zero for an effect (``AGENT_REVISION_PROTOCOL.md``: "the group-clustered
  95% interval ... is wholly above zero");
- rank correlation reuses the project's average-rank Spearman helpers in
  ``codex_hydrogym.genai.metrics``.

Arms: the null arm draws two groups from the SAME quality tier (differing
only by random seed); the positive arms draw groups from deliberately
different tiers. A correct instrument reports no effect on the null arm and
an effect, with recovered ordering, on the positive arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Protocol

from codex_hydrogym.agent_eval.instrument_validation.generator import (
    QUALITY_TIERS,
    RewardCandidate,
    RewardCandidateGenerator,
)
from codex_hydrogym.agent_eval.instrument_validation.judges import (
    CeilingPinnedJudge,
    DeterministicTierJudge,
)
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci
from codex_hydrogym.gate0.ensemble_replication import EnsembleReplicationSpec
from codex_hydrogym.genai.metrics import _average_ranks, _pearson

GROUP_CLUSTERS: int = 10
T_CRITICAL_95_DF9: float = EnsembleReplicationSpec.seed_cluster_t_critical_95
NULL_ARM_TIER: int = 2
POSITIVE_ARM_TIER_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 1),
    (3, 4),
)
SEED_NAMESPACE: int = 20260826
DECISION_EFFECT = "effect"
DECISION_NO_EFFECT = "no_effect"


class Judge(Protocol):
    name: str

    def score(self, candidate: RewardCandidate) -> float: ...


@dataclass(frozen=True)
class PairedArmResult:
    """Outcome of one paired control run through the instrument."""

    arm_name: str
    control_tier: int
    candidate_tier: int
    group_clusters: int
    judge_name: str
    paired_mean_delta: float
    paired_deltas: Mapping[str, float]
    clustered_interval_95: Mapping[str, float]
    decision: str

    def __post_init__(self) -> None:
        if not isinstance(self.arm_name, str) or not self.arm_name.strip():
            raise ValueError("arm_name must be non-empty")
        if self.control_tier not in QUALITY_TIERS or self.candidate_tier not in QUALITY_TIERS:
            raise ValueError("arm tiers must lie on the constructed quality ladder")
        if self.group_clusters < 2:
            raise ValueError("a paired arm needs at least two group clusters")
        if len(self.paired_deltas) != self.group_clusters:
            raise ValueError("paired deltas must cover exactly the group clusters")
        if self.decision not in {DECISION_EFFECT, DECISION_NO_EFFECT}:
            raise ValueError("decision must be one of effect / no_effect")


@dataclass(frozen=True)
class OrderingResult:
    """Recovered ordering of tier-mean judge scores against the known ladder."""

    tier_scores: Mapping[int, float]
    spearman: float

    def __post_init__(self) -> None:
        if set(self.tier_scores) != set(QUALITY_TIERS):
            raise ValueError("ordering must cover every constructed tier")


@dataclass(frozen=True)
class ValidationResult:
    """Full instrument-validation run: one null arm, positive arms, ordering."""

    null_arm: PairedArmResult
    positive_arms: tuple[PairedArmResult, ...]
    ordering: OrderingResult
    judge_name: str

    def __post_init__(self) -> None:
        if self.null_arm.control_tier != self.null_arm.candidate_tier:
            raise ValueError("the null arm must draw both groups from one tier")
        if not self.positive_arms:
            raise ValueError("at least one positive arm is required")
        if any(arm.control_tier == arm.candidate_tier for arm in self.positive_arms):
            raise ValueError("positive arms must pair deliberately different tiers")


def _paired_arm(
    *,
    arm_name: str,
    control_tier: int,
    candidate_tier: int,
    judge: Judge,
    seed_a: int,
    seed_b: int,
    groups: int,
) -> PairedArmResult:
    control = RewardCandidateGenerator(tier=control_tier, groups=groups, seed=seed_a).generate()
    candidate = RewardCandidateGenerator(tier=candidate_tier, groups=groups, seed=seed_b).generate()
    control_by_group = {item.group_id: item for item in control}
    candidate_by_group = {item.group_id: item for item in candidate}
    if control_by_group.keys() != candidate_by_group.keys():
        raise AssertionError("paired arms must expose the same group clusters")
    # Frozen protocol expression: paired delta per group, then the clustered mean.
    deltas = {
        group_id: judge.score(candidate_by_group[group_id]) - judge.score(control_by_group[group_id])
        for group_id in control_by_group
    }
    interval = _mean_ci(tuple(deltas.values()), T_CRITICAL_95_DF9)
    decision = DECISION_EFFECT if interval["lower"] > 0.0 else DECISION_NO_EFFECT
    return PairedArmResult(
        arm_name=arm_name,
        control_tier=control_tier,
        candidate_tier=candidate_tier,
        group_clusters=groups,
        judge_name=judge.name,
        paired_mean_delta=fmean(deltas.values()),
        paired_deltas=dict(deltas),
        clustered_interval_95=dict(interval),
        decision=decision,
    )


def _ordering(judge: Judge, *, seed: int, groups: int) -> OrderingResult:
    tier_scores = {
        tier: fmean(
            judge.score(candidate)
            for candidate in RewardCandidateGenerator(tier=tier, groups=groups, seed=seed).generate()
        )
        for tier in QUALITY_TIERS
    }
    tiers = [float(tier) for tier in QUALITY_TIERS]
    scores = [tier_scores[tier] for tier in QUALITY_TIERS]
    spearman = _pearson(_average_ranks(tiers), _average_ranks(scores))
    if spearman is None:
        raise AssertionError("ordering spearman is undefined; scores must vary with tier")
    return OrderingResult(tier_scores=tier_scores, spearman=spearman)


def run_validation(
    judge: Judge | None = None,
    *,
    seed: int = SEED_NAMESPACE,
    groups: int = GROUP_CLUSTERS,
) -> ValidationResult:
    """Run the null arm, all positive arms, and the ordering check."""
    judge = judge or DeterministicTierJudge()
    if not isinstance(groups, int) or groups < 2:
        raise ValueError("groups must be an int of at least two clusters")
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    null_arm = _paired_arm(
        arm_name="null",
        control_tier=NULL_ARM_TIER,
        candidate_tier=NULL_ARM_TIER,
        judge=judge,
        seed_a=seed,
        seed_b=seed + 1,
        groups=groups,
    )
    positive_arms = tuple(
        _paired_arm(
            arm_name=f"positive-{control_tier}-{candidate_tier}",
            control_tier=control_tier,
            candidate_tier=candidate_tier,
            judge=judge,
            seed_a=seed + 10 + 2 * index,
            seed_b=seed + 10 + 2 * index + 1,
            groups=groups,
        )
        for index, (control_tier, candidate_tier) in enumerate(POSITIVE_ARM_TIER_PAIRS)
    )
    ordering = _ordering(judge, seed=seed + 100, groups=groups)
    return ValidationResult(
        null_arm=null_arm,
        positive_arms=positive_arms,
        ordering=ordering,
        judge_name=judge.name,
    )


def run_ceiling_pinned_reference(*, seed: int = SEED_NAMESPACE, groups: int = GROUP_CLUSTERS) -> PairedArmResult:
    """Reference: the ceiling-pinned judge against a real tier gap.

    Documents the current production symptom (every paired delta exactly 0.0)
    and proves the statistic reports no effect when the judge carries no
    signal, even though the underlying tiers differ as much as possible.
    """
    if not isinstance(groups, int) or groups < 2:
        raise ValueError("groups must be an int of at least two clusters")
    return _paired_arm(
        arm_name="ceiling-pinned-reference",
        control_tier=0,
        candidate_tier=4,
        judge=CeilingPinnedJudge(),
        seed_a=seed + 200,
        seed_b=seed + 201,
        groups=groups,
    )


def summarize(result: ValidationResult) -> dict[str, Any]:
    """Human-readable summary of a validation run (for reports, not claims)."""
    return {
        "judge_name": result.judge_name,
        "null_arm": {
            "tier": result.null_arm.control_tier,
            "paired_mean_delta": result.null_arm.paired_mean_delta,
            "clustered_interval_95": dict(result.null_arm.clustered_interval_95),
            "decision": result.null_arm.decision,
        },
        "positive_arms": [
            {
                "control_tier": arm.control_tier,
                "candidate_tier": arm.candidate_tier,
                "paired_mean_delta": arm.paired_mean_delta,
                "clustered_interval_95": dict(arm.clustered_interval_95),
                "decision": arm.decision,
            }
            for arm in result.positive_arms
        ],
        "ordering_spearman": result.ordering.spearman,
        "tier_scores": dict(result.ordering.tier_scores),
    }

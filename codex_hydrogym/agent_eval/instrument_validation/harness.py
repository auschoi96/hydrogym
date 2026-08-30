"""Seeded calibration of the preregistered group-clustered interval rule.

This validates only the decision rule preregistered at
``agent_eval/AGENT_REVISION_PROTOCOL.md:45``: a group-clustered 95% interval
wholly above zero. It does not validate the notebook's currently executed bare
mean-sign decision (``notebooks/coding_agent_memalign_proof.py:368-376``).

The stochastic DGP is implemented by :class:`SeededNoisyTierJudge`: independent
``Normal(0, sigma**2)`` noise is added to the known tier mean. Every arm uses
``CALIBRATION_REPLICATES`` explicit consecutive seeds. The default sigma,
replicate count, monotonicity tolerance, and ordering threshold are fixed below
and are not selected from results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Mapping, Protocol

from codex_hydrogym.agent_eval.instrument_validation.generator import (
    QUALITY_TIERS,
    TIER_BASE_SCORE,
    RewardCandidate,
    RewardCandidateGenerator,
)
from codex_hydrogym.agent_eval.instrument_validation.judges import (
    CeilingPinnedJudge,
    DeterministicTierJudge,
    SeededNoisyTierJudge,
)
from codex_hydrogym.gate0.ensemble_diagnostic import _mean_ci
from codex_hydrogym.gate0.ensemble_replication import EnsembleReplicationSpec
from codex_hydrogym.genai.metrics import _average_ranks, _pearson

GROUP_CLUSTERS = 10
T_CRITICAL_95_DF9 = EnsembleReplicationSpec.seed_cluster_t_critical_95
NULL_ARM_TIER = 2
POSITIVE_ARM_TIER_PAIRS = ((0, 1), (1, 3), (2, 4), (3, 4))
SEED_NAMESPACE = 20260826
CALIBRATION_REPLICATES = 500
NOISE_SIGMA = 1.0
MONOTONICITY_TOLERANCE = 0.10
MIN_MEAN_NOISY_SPEARMAN = 0.90
DECISION_EFFECT = "effect"
DECISION_NO_EFFECT = "no_effect"


class Judge(Protocol):
    name: str

    def score(self, candidate: RewardCandidate) -> float: ...


@dataclass(frozen=True)
class PairedArmResult:
    arm_name: str
    control_tier: int
    candidate_tier: int
    group_clusters: int
    judge_name: str
    paired_mean_delta: float
    paired_deltas: Mapping[str, float]
    clustered_interval_95: Mapping[str, float]
    t_critical_95: float
    t_degrees_of_freedom: int
    decision: str


@dataclass(frozen=True)
class OrderingResult:
    tier_scores: Mapping[int, float]
    spearman: float


@dataclass(frozen=True)
class ValidationResult:
    null_arm: PairedArmResult
    positive_arms: tuple[PairedArmResult, ...]
    ordering: OrderingResult
    judge_name: str


@dataclass(frozen=True)
class TierPairCalibration:
    control_tier: int
    candidate_tier: int
    true_delta: float
    coverage: float
    detection_rate: float


@dataclass(frozen=True)
class CalibrationResult:
    replicates: int
    seeds: tuple[int, ...]
    sigma: float
    null_false_positive_rate: float
    null_false_positive_interval_95: Mapping[str, float]
    positive_coverage: float
    positive_coverage_interval_95: Mapping[str, float]
    positive_detection_rate: float
    tier_pairs: tuple[TierPairCalibration, ...]
    mean_ordering_spearman: float


def _validate_groups(groups: int) -> None:
    # The frozen source constant is 2.262157..., documented specifically as
    # ten clusters / df=9 at ensemble_replication.py:154-157.
    if groups != GROUP_CLUSTERS:
        raise ValueError("groups must be exactly 10: the frozen t critical is for df=9")


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
    _validate_groups(groups)
    control = RewardCandidateGenerator(tier=control_tier, groups=groups, seed=seed_a).generate()
    candidate = RewardCandidateGenerator(tier=candidate_tier, groups=groups, seed=seed_b).generate()
    control_by_group = {item.group_id: item for item in control}
    candidate_by_group = {item.group_id: item for item in candidate}
    if control_by_group.keys() != candidate_by_group.keys():
        raise AssertionError("paired arms must expose the same group clusters")
    deltas = {}
    for group_id in control_by_group:
        # Frozen notebook expression (lines 355-364), with identical treatment
        # labels. test_paired_delta_expression_is_pinned_to_notebook AST-pins it.
        by_treatment = {
            "unchanged": judge.score(control_by_group[group_id]),
            "base_revision": judge.score(candidate_by_group[group_id]),
        }
        deltas[group_id] = by_treatment["base_revision"] - by_treatment["unchanged"]
    interval = _mean_ci(tuple(deltas.values()), T_CRITICAL_95_DF9)
    return PairedArmResult(
        arm_name=arm_name,
        control_tier=control_tier,
        candidate_tier=candidate_tier,
        group_clusters=groups,
        judge_name=judge.name,
        paired_mean_delta=fmean(deltas.values()),
        paired_deltas=deltas,
        clustered_interval_95=dict(interval),
        t_critical_95=T_CRITICAL_95_DF9,
        t_degrees_of_freedom=groups - 1,
        decision=DECISION_EFFECT if interval["lower"] > 0 else DECISION_NO_EFFECT,
    )


def _ordering(judge: Judge, *, seed: int, groups: int) -> OrderingResult:
    tier_scores = {
        tier: fmean(
            judge.score(candidate)
            for candidate in RewardCandidateGenerator(tier=tier, groups=groups, seed=seed + tier).generate()
        )
        for tier in QUALITY_TIERS
    }
    ranks = _pearson(
        _average_ranks([float(tier) for tier in QUALITY_TIERS]),
        _average_ranks([tier_scores[tier] for tier in QUALITY_TIERS]),
    )
    if ranks is None:
        raise AssertionError("ordering spearman is undefined")
    return OrderingResult(tier_scores=tier_scores, spearman=ranks)


def run_validation(
    judge: Judge | None = None, *, seed: int = SEED_NAMESPACE, groups: int = GROUP_CLUSTERS
) -> ValidationResult:
    """Run one diagnostic replicate; calibration claims use ``run_calibration``."""
    _validate_groups(groups)
    if not isinstance(seed, int):
        raise TypeError("seed must be an int")
    judge = judge or DeterministicTierJudge()
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
            arm_name=f"positive-{a}-{b}",
            control_tier=a,
            candidate_tier=b,
            judge=judge,
            seed_a=seed + 10 + 2 * index,
            seed_b=seed + 11 + 2 * index,
            groups=groups,
        )
        for index, (a, b) in enumerate(POSITIVE_ARM_TIER_PAIRS)
    )
    return ValidationResult(null_arm, positive_arms, _ordering(judge, seed=seed + 100, groups=groups), judge.name)


def _wilson_interval(successes: int, trials: int) -> dict[str, float]:
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {"lower": center - radius, "upper": center + radius}


def run_calibration(
    *, replicates: int = CALIBRATION_REPLICATES, seed: int = SEED_NAMESPACE, sigma: float = NOISE_SIGMA
) -> CalibrationResult:
    """Measure null FPR, positive coverage/detection, sensitivity, and ordering."""
    if replicates < 200:
        raise ValueError("calibration requires at least 200 seeded replicates")
    seeds = tuple(seed + replicate for replicate in range(replicates))
    null_false_positives = 0
    coverages = {pair: 0 for pair in POSITIVE_ARM_TIER_PAIRS}
    detections = {pair: 0 for pair in POSITIVE_ARM_TIER_PAIRS}
    spearman = []
    for replicate_seed in seeds:
        result = run_validation(SeededNoisyTierJudge(seed=replicate_seed, sigma=sigma), seed=replicate_seed)
        all_arms = (result.null_arm, *result.positive_arms)
        if any(arm.clustered_interval_95["lower"] >= arm.clustered_interval_95["upper"] for arm in all_arms):
            raise AssertionError("every stochastic interval must have positive width")
        null_false_positives += result.null_arm.decision == DECISION_EFFECT
        for arm in result.positive_arms:
            pair = (arm.control_tier, arm.candidate_tier)
            true_delta = TIER_BASE_SCORE[pair[1]] - TIER_BASE_SCORE[pair[0]]
            interval = arm.clustered_interval_95
            coverages[pair] += interval["lower"] <= true_delta <= interval["upper"]
            detections[pair] += arm.decision == DECISION_EFFECT
        spearman.append(result.ordering.spearman)
    pair_results = tuple(
        TierPairCalibration(
            a,
            b,
            TIER_BASE_SCORE[b] - TIER_BASE_SCORE[a],
            coverages[(a, b)] / replicates,
            detections[(a, b)] / replicates,
        )
        for a, b in POSITIVE_ARM_TIER_PAIRS
    )
    # Pair-level sensitivity may vary by Monte Carlo error. A fixed 10-point
    # tolerance permits that error but still requires larger gaps to detect at
    # least as often as every smaller gap.
    for small in pair_results:
        for large in pair_results:
            if large.true_delta > small.true_delta:
                if large.detection_rate + MONOTONICITY_TOLERANCE < small.detection_rate:
                    raise AssertionError("detection sensitivity is not monotone within tolerance")
    mean_spearman = fmean(spearman)
    if mean_spearman < MIN_MEAN_NOISY_SPEARMAN:
        raise AssertionError("noisy ordering fell below the fixed Spearman tolerance")
    total_positive = replicates * len(pair_results)
    return CalibrationResult(
        replicates=replicates,
        seeds=seeds,
        sigma=float(sigma),
        null_false_positive_rate=null_false_positives / replicates,
        null_false_positive_interval_95=_wilson_interval(null_false_positives, replicates),
        positive_coverage=sum(coverages.values()) / total_positive,
        positive_coverage_interval_95=_wilson_interval(sum(coverages.values()), total_positive),
        positive_detection_rate=sum(detections.values()) / total_positive,
        tier_pairs=pair_results,
        mean_ordering_spearman=mean_spearman,
    )


def run_ceiling_pinned_reference(*, seed: int = SEED_NAMESPACE, groups: int = GROUP_CLUSTERS) -> PairedArmResult:
    """Dead-judge reference, explicitly not a stochastic null calibration."""
    return _paired_arm(
        arm_name="ceiling-pinned-reference",
        control_tier=0,
        candidate_tier=4,
        judge=CeilingPinnedJudge(),
        seed_a=seed + 200,
        seed_b=seed + 201,
        groups=groups,
    )


def summarize(result: CalibrationResult) -> dict[str, Any]:
    """Return the measured calibration artifact in JSON-compatible form."""
    return {
        "replicates": result.replicates,
        "seed_first": result.seeds[0],
        "seed_last": result.seeds[-1],
        "sigma": result.sigma,
        "null_false_positive_rate": result.null_false_positive_rate,
        "null_false_positive_interval_95": dict(result.null_false_positive_interval_95),
        "positive_coverage": result.positive_coverage,
        "positive_coverage_interval_95": dict(result.positive_coverage_interval_95),
        "positive_detection_rate": result.positive_detection_rate,
        "tier_pairs": [vars(item) for item in result.tier_pairs],
        "mean_ordering_spearman": result.mean_ordering_spearman,
    }

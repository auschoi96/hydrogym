"""Judge stubs for paired-delta instrument validation."""

from __future__ import annotations

import random

from codex_hydrogym.agent_eval.instrument_validation.generator import (
    QUALITY_TIERS,
    TIER_BASE_SCORE,
    TIER_SIGNATURES,
    RewardCandidate,
)


def _resolved_tier(candidate: RewardCandidate) -> int:
    if not isinstance(candidate, RewardCandidate):
        raise TypeError("score requires a RewardCandidate")
    matched = [tier for tier in QUALITY_TIERS if all(stem in candidate.text for stem in TIER_SIGNATURES[tier])]
    if len(matched) != 1 or matched[0] != candidate.tier:
        raise AssertionError("judge did not resolve exactly the constructed tier")
    return matched[0]


class DeterministicTierJudge:
    """Exact-monotone diagnostic stub; not calibration evidence."""

    name = "deterministic_tier_stub"

    def score(self, candidate: RewardCandidate) -> float:
        return TIER_BASE_SCORE[_resolved_tier(candidate)]


class SeededNoisyTierJudge:
    """Seeded stochastic judge with an explicit data-generating process.

    For candidate ``i`` in tier ``t``, independently generate
    ``score_i = TIER_BASE_SCORE[t] + epsilon_i`` where
    ``epsilon_i ~ Normal(0, sigma**2)``. Independence and reproducibility come
    from a local RNG keyed by the explicit judge seed and candidate seed. The
    default ``sigma=1.0`` is preregistered in the harness constants rather than
    selected from experiment output. Scores are intentionally not clipped:
    clipping would alter the stated normal DGP and tier mean differences.
    """

    name = "seeded_noisy_tier_stub"

    def __init__(self, *, seed: int, sigma: float) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        if not isinstance(sigma, (float, int)) or sigma <= 0:
            raise ValueError("sigma must be positive")
        self.seed = seed
        self.sigma = float(sigma)

    def score(self, candidate: RewardCandidate) -> float:
        tier = _resolved_tier(candidate)
        rng = random.Random(f"instrument-validation:{self.seed}:{candidate.seed}")
        return TIER_BASE_SCORE[tier] + rng.gauss(0.0, self.sigma)


class CeilingPinnedJudge:
    """Dead-judge reference: all candidates receive the same ceiling score."""

    name = "ceiling_pinned_stub"

    def __init__(self, ceiling: float = 5.0) -> None:
        if not 1.0 <= ceiling <= 5.0:
            raise ValueError("ceiling must lie on the critic scale [1, 5]")
        self.ceiling = ceiling

    def score(self, candidate: RewardCandidate) -> float:
        if not isinstance(candidate, RewardCandidate):
            raise TypeError("score requires a RewardCandidate")
        return self.ceiling

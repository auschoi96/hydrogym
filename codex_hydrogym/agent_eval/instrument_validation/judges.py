"""Deterministic judge stubs for instrument validation.

The LLM judge is mocked with a deterministic stub, never the statistic under
test. Assumption, stated explicitly: a candidate's score is a pure function of
its constructed quality tier, monotone non-decreasing in the known tier, with
zero within-tier variance. The generator emits tier-exclusive signature
phrases, and the stub reads exactly those phrases off the candidate text, so
the stub is exactly monotone: any tier-``k`` candidate scores strictly above
any tier-``j < k`` candidate, and any two same-tier candidates score
identically regardless of their random seeds. A stochastic LLM judge would add
within-tier noise; this validation deliberately measures the paired-delta
statistic pipeline under a noise-free oracle so the null arm has a true delta
of exactly zero.
"""

from __future__ import annotations

from codex_hydrogym.agent_eval.instrument_validation.generator import (
    QUALITY_TIERS,
    TIER_BASE_SCORE,
    TIER_SIGNATURES,
    RewardCandidate,
)


class DeterministicTierJudge:
    """Exact-monotone stub judge on the project's 1-5 critic scale."""

    name = "deterministic_tier_stub"

    def score(self, candidate: RewardCandidate) -> float:
        if not isinstance(candidate, RewardCandidate):
            raise TypeError("score requires a RewardCandidate")
        matched = [tier for tier in QUALITY_TIERS if all(stem in candidate.text for stem in TIER_SIGNATURES[tier])]
        if len(matched) != 1 or matched[0] != candidate.tier:
            raise AssertionError("judge did not resolve exactly the constructed tier")
        return TIER_BASE_SCORE[matched[0]]


class CeilingPinnedJudge:
    """Reference replica of the current production symptom.

    Every candidate scores the ceiling value 5.0, which is exactly how the
    production judge behaves today (5.0 -> 5.0, 0.9 -> 0.9, 1.0 -> 1.0). This
    stub is deliberately NOT monotone in the quality tier; it exists only to
    show what the paired-delta instrument reports when the judge carries no
    signal.
    """

    name = "ceiling_pinned_stub"

    def __init__(self, ceiling: float = 5.0) -> None:
        if not 1.0 <= ceiling <= 5.0:
            raise ValueError("ceiling must lie on the critic scale [1, 5]")
        self.ceiling = ceiling

    def score(self, candidate: RewardCandidate) -> float:
        if not isinstance(candidate, RewardCandidate):
            raise TypeError("score requires a RewardCandidate")
        return self.ceiling

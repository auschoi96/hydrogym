"""Instrument validation: null and positive controls for the paired-delta harness.

This package validates the measurement instrument, not a coding-agent
treatment. It proves that the paired-delta statistic the project already uses
can (a) report no effect when none exists and (b) detect a known, constructed
effect when one exists. It produces no claim about coding agents improving
control, and it never executes CFD or RL training.
"""

from codex_hydrogym.agent_eval.instrument_validation.generator import (
    QUALITY_TIERS,
    TIER_BASE_SCORE,
    RewardCandidate,
    RewardCandidateGenerator,
)
from codex_hydrogym.agent_eval.instrument_validation.harness import (
    DECISION_EFFECT,
    DECISION_NO_EFFECT,
    GROUP_CLUSTERS,
    POSITIVE_ARM_TIER_PAIRS,
    PairedArmResult,
    ValidationResult,
    run_ceiling_pinned_reference,
    run_validation,
)
from codex_hydrogym.agent_eval.instrument_validation.judges import (
    CeilingPinnedJudge,
    DeterministicTierJudge,
)

__all__ = [
    "CeilingPinnedJudge",
    "DECISION_EFFECT",
    "DECISION_NO_EFFECT",
    "DeterministicTierJudge",
    "GROUP_CLUSTERS",
    "POSITIVE_ARM_TIER_PAIRS",
    "QUALITY_TIERS",
    "PairedArmResult",
    "RewardCandidate",
    "RewardCandidateGenerator",
    "TIER_BASE_SCORE",
    "ValidationResult",
    "run_ceiling_pinned_reference",
    "run_validation",
]

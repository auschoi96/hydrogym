"""Implementation namespace for work labeled codex_hydrogym."""

PROJECT_LABEL = "codex_hydrogym"
PROJECT_TAG = "codex_hydrogym.project"
LEGACY_REWARD_FORMULA_VERSION = "hydrogym.tke_alpha_l1.v1"
REWARD_FORMULA_VERSION = "codex_hydrogym.normalized_tke_l1_delta.v1"
FEEDBACK_ASSESSMENT_NAME = "fluid_reward_plausibility"
CRITIC_QUALITY_ASSESSMENT_NAME = "critic_quality"

__all__ = [
    "CRITIC_QUALITY_ASSESSMENT_NAME",
    "FEEDBACK_ASSESSMENT_NAME",
    "LEGACY_REWARD_FORMULA_VERSION",
    "PROJECT_LABEL",
    "PROJECT_TAG",
    "REWARD_FORMULA_VERSION",
]

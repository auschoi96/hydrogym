"""MemAlign H1: held-out judge-agreement harness, name-match preflight, locked labeling folds.

The only MemAlign claim that is affordably measurable here is H1: after alignment on a
locked TRAIN fold of adjudicated HUMAN ``critic_quality`` labels, does the aligned
reviewer agree with HELD-OUT HUMAN labels better than the base reviewer?

The central risk is a judge-name collision.  Three names exist in this repository:

- ``fluid_reward_plausibility``: ``FEEDBACK_ASSESSMENT_NAME``
  (codex_hydrogym/__init__.py:7) and ``FEEDBACK_NAME`` in the labeling App
  (codex_hydrogym/app/backend.py:19) -- this is what the App's labeling path writes;
- ``critic_quality``: ``CRITIC_QUALITY_ASSESSMENT_NAME`` (codex_hydrogym/__init__.py:8) --
  the adjudication pipeline built around codex_hydrogym/genai/feedback.py and
  codex_hydrogym/genai/optimization.py::align_critic_quality_judge, which is the H1
  alignment target;
- ``codex_hydrogym_revision_audit_v1``: ``AUDIT_JUDGE_NAME``
  (notebooks/coding_agent_memalign_proof.py:69) -- registered outcome auditor whose
  assessments are produced by a model (LLM_JUDGE source).

MemAlign ``align(judge, traces)`` keeps only assessments whose sanitized name matches the
sanitized judge name AND whose source type is HUMAN (trace_to_dspy_example,
dspy_utils.py:391-396); everything else is silently dropped, and align then raises
"No valid feedback records found in traces." Labels written under one name are invisible
to a judge named another.
"""

from __future__ import annotations

from codex_hydrogym import CRITIC_QUALITY_ASSESSMENT_NAME, FEEDBACK_ASSESSMENT_NAME

PROTOCOL_ID = "codex_hydrogym.memalign_h1.v1"
PROTOCOL_PATH = "codex_hydrogym/agent_eval/MEMALIGN_H1_PROTOCOL.md"

# The registered outcome auditor used by the revision study.  Its model-produced
# assessments are LLM_JUDGE-sourced, so it is not alignable while it has no HUMAN
# assessments -- preflight makes this visible instead of failing late in MemAlign.
AUDIT_JUDGE_NAME = "codex_hydrogym_revision_audit_v1"

# Every judge name H1 must know about.  A label written under one name is invisible to a
# judge named another: this is the collision the preflight must surface.
MEMALIGN_JUDGE_NAMES = (FEEDBACK_ASSESSMENT_NAME, CRITIC_QUALITY_ASSESSMENT_NAME, AUDIT_JUDGE_NAME)

# Frozen 95% t-critical for the group-clustered delta-MAE interval.  This is the exact
# value frozen in codex_hydrogym/gate0/ensemble_diagnostic.py:112
# (EnsembleDiagnosticSpec.seed_cluster_t_critical_95) for four clusters (df=3).  H1
# reuses that frozen critical value, so the held-out fold must contain exactly four
# group clusters.
FROZEN_T_CRITICAL_95 = 3.182446305284263
FROZEN_HELDOUT_GROUP_COUNT = 4

# Required adjudicated HUMAN critic_quality labels before H1 can be decided.  The train
# fold needs at least 50 records -- exactly mlflow's MemAlign distillation batch cap
# (_MAX_RECORDS_PER_BATCH, memalign/utils.py:41).  The held-out fold needs 4 groups x 2
# harness arms (codex, claude) = 8 labels.  No eligible HUMAN label exists today.
REQUIRED_TRAIN_LABEL_COUNT = 50
REQUIRED_HELDOUT_LABEL_COUNT = 8
REQUIRED_TOTAL_LABEL_COUNT = REQUIRED_TRAIN_LABEL_COUNT + REQUIRED_HELDOUT_LABEL_COUNT

# Salt for the deterministic SHA-256 group ranking reused from grouped_bundle_split, and
# the fixed seed for the secondary group-clustered bootstrap interval.
H1_SPLIT_SALT = "codex_hydrogym.memalign_h1.fold.v1"
FROZEN_BOOTSTRAP_SEED = 7021
FROZEN_BOOTSTRAP_REPLICATES = 2000

__all__ = [
    "AUDIT_JUDGE_NAME",
    "CRITIC_QUALITY_ASSESSMENT_NAME",
    "FEEDBACK_ASSESSMENT_NAME",
    "FROZEN_BOOTSTRAP_REPLICATES",
    "FROZEN_BOOTSTRAP_SEED",
    "FROZEN_HELDOUT_GROUP_COUNT",
    "FROZEN_T_CRITICAL_95",
    "H1_SPLIT_SALT",
    "MEMALIGN_JUDGE_NAMES",
    "PROTOCOL_ID",
    "PROTOCOL_PATH",
    "REQUIRED_HELDOUT_LABEL_COUNT",
    "REQUIRED_TOTAL_LABEL_COUNT",
    "REQUIRED_TRAIN_LABEL_COUNT",
]

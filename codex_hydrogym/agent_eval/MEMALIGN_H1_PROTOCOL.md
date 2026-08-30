# MemAlign H1: held-out judge-agreement protocol

**Frozen:** with this commit, before any alignment, enrollment, or label collection
**Protocol ID:** `codex_hydrogym.memalign_h1.v1`
**Judge target:** the registered `critic_quality` reviewer
(`CRITIC_QUALITY_ASSESSMENT_NAME = "critic_quality"`, `codex_hydrogym/__init__.py:8`)
**Label source:** adjudicated HUMAN `critic_quality` assessments only
(`codex_hydrogym.genai.feedback.submit_adjudicated_critic_quality`)

## Question (H1)

After MemAlign alignment on a locked TRAIN fold of adjudicated HUMAN `critic_quality`
labels, does the aligned `critic_quality` reviewer agree with HELD-OUT HUMAN
`critic_quality` labels better than the base reviewer, on mean absolute error (MAE)?

This is the only MemAlign claim this harness can affordably measure. It is an
agent-quality claim about judge agreement. It is not an RL claim and not a fluid claim.

## Null hypothesis

MemAlign alignment (episodic memory + retrieval_k=5 + guideline distillation; no
gradient training) produces no change in agreement with held-out HUMAN labels:
delta-MAE = 0, where delta-MAE = MAE(aligned) − MAE(base). A negative delta-MAE favors
the aligned judge.

## Required labels before H1 can be decided

No eligible HUMAN assessment exists today (MemAlign has never run; preflight shows zero
surviving assessments for every judge name). Before an H1 decision:

1. at least **50** adjudicated `critic_quality` HUMAN labels on the locked TRAIN fold —
   exactly MLflow MemAlign's distillation batch cap (`_MAX_RECORDS_PER_BATCH`,
   `memalign/utils.py:41`), distributed over groups so that no group straddles folds;
2. exactly **8** held-out labels on the HELD-OUT fold — 4 group clusters × 2 harness
   arms (`codex`, `claude`). The held-out fold must contain exactly **4** group
   clusters because the 95% t-critical is frozen to the df=3 value used by
   `codex_hydrogym.gate0.ensemble_diagnostic`
   (`EnsembleDiagnosticSpec.seed_cluster_t_critical_95 = 3.182446305284263`, 4 clusters).

Total: **58** adjudicated `critic_quality` HUMAN labels (50 train + 8 held-out).
Labels are collected only from the frozen manifest emitted by the labeling queue; the
manifest SHA-256 is fixed BEFORE the first label is collected, and enrollment refuses
any mutated manifest.

## Fold structure

- Folds are assigned to whole `group_id` values, never to individual traces, using the
  repository's own SHA-256 group ranking (`grouped_bundle_split`,
  `codex_hydrogym/genai/datasets.py:340`) with the H1 split salt
  `codex_hydrogym.memalign_h1.fold.v1`.
- Group-disjoint means no group appears in both folds; the emptiness and
  disjoint-manifest checks fail loudly (locking reuses
  `align_critic_quality_judge`, which raises on any trace outside the locked train
  manifest, on any held-out bundle inside it, and on any train trace without exactly
  one adjudicated HUMAN label).
- A trace whose ONLY `critic_quality` assessment is machine-produced
  (source_type ≠ HUMAN) is rejected by the same guard; no synthetic, LLM-generated,
  or assistant-generated assessment may ever be recorded with source_type=HUMAN.
- The PPO repair-task traces from `coding_rl/experiment.py` (tagged `task_id`,
  `condition`, `critic_fold`) belong to the repair-task experiment and carry no
  canonical RunBundle in their root span; they are excluded from the critic_quality
  queue with an explicit reason and never enter H1 folds.

## Alignment

- Alignment uses ONLY locked-TRAIN traces: `align_critic_quality_judge`
  (`codex_hydrogym/genai/optimization.py:143`) with retrieval_k=5. Held-out labels are
  unreachable during alignment: the derived `evaluate_h1` harness passes only train
  traces and train bundle ids to the aligner, and the aligner itself rejects held-out
  bundles in the training set.
- The aligned reviewer is a separately versioned scorer; the base `critic_quality`
  reviewer is never overwritten.

## Decision rule

The primary endpoint is the group-clustered 95% interval on delta-MAE over the
HELD-OUT fold. One delta-MAE per group cluster; the interval is
`mean ± t_critical × SE` computed with `_mean_ci`
(`codex_hydrogym/gate0/ensemble_diagnostic.py:381`) and the frozen df=3 t-critical
(`3.182446305284263`), exactly as Gate 0's ensemble diagnostic computes its
seed-cluster intervals. A fixed-seed (`7021`) group-clustered bootstrap interval (2000
replicates, whole-group resampling) is reported as a secondary sensitivity statistic
only and never carries the decision.

- **PASS** only when the held-out delta-MAE 95% interval is wholly favorable
  (upper bound < 0: the aligned judge agrees better with held-out HUMAN labels).
- **FAIL** when the interval is wholly unfavorable (lower bound > 0: the aligned judge
  regressed). Regressions on any dimension are reported, not hidden.
- **INCONCLUSIVE** when the interval straddles zero, or when the required label counts
  do not exist yet. A negative H1 (FAIL or INCONCLUSIVE) is a legitimate result; there
  is no tuning to pass.

Per-dimension MAE for base and aligned judges is reported for the label schema's
numeric dimension (`critic_quality`; the analysis is defined over D dimensions).

## Collision (why preflight exists)

MemAlign keeps only assessments whose sanitized name matches the sanitized judge name
AND whose source is HUMAN (`trace_to_dspy_example`, `dspy_utils.py:391-396`); if none
survive it raises "No valid feedback records found in traces."
(`optimizer.py:656-663`). The labeling App writes `fluid_reward_plausibility`
(`codex_hydrogym/app/backend.py:19`); the adjudication pipeline and this protocol
target `critic_quality`; the outcome auditor `codex_hydrogym_revision_audit_v1` is
model-scored. Labels under one name are invisible to a judge named another. The
name-match preflight (`codex_hydrogym.memalign_h1.preflight`) must report, for each
judge name, exactly how many assessments would survive and why each rejection happened
(name mismatch vs non-HUMAN source) before any alignment run.

## What a PASS does NOT license

A PASS licenses exactly one sentence: alignment improved held-out agreement with HUMAN
`critic_quality` labels in this study. It licenses NO claim about:

- reinforcement learning or the PPO trainer: the PPO reward is deterministic
  (`coding_rl/experiment.py:1110-1114`) and MemAlign is not in that path;
- fluid performance, controller quality, TKE, or control effort;
- `fluid_reward_plausibility`, GEPA, prompt promotion, or model promotion;
- any judge score as proof of control improvement.

## Forbidden actions and claims

- no synthetic, LLM-generated, or assistant-generated assessment may be labeled HUMAN;
- no opened held-out label may be used for alignment, case selection, prompt revision,
  or stopping;
- no judge may score the outcome of advice it produced;
- no modification of `codex_hydrogym/gate0/` and no weakening of existing tests;
- no claim that H1 results generalize beyond the exact frozen manifest and protocol id.

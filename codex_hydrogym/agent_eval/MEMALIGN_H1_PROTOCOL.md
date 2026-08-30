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

## Label budget and power analysis

No eligible HUMAN assessment or pilot variance estimate exists today. Submission
mechanics and statistical sufficiency are separate:

- MLflow's `_MAX_RECORDS_PER_BATCH = 50` is only a **batching cap**. It says nothing
  about sample sufficiency.
- **50 TRAIN labels is a pilot guess**, not a powered training count. There is no real
  MemAlign learning curve yet. A repeated group-resampled learning curve over the real
  optimizer can only be run after labels exist; until then stabilization is unknown.
- The proposed **8 held-out labels** are 4 groups × 2 arms. Four groups are an
  underpowered pilot, not a confirmatory design.

`python -m codex_hydrogym.memalign_h1.power` is the preregisterable, seed-7021
simulation. It applies the exact PASS/FAIL/INCONCLUSIVE rule to independent group
level delta-MAEs, sweeps normal and standardized heavy-tailed t(5) distributions,
group SDs 0.25/0.50/0.75, improvements 0 through 1.0, and unequal group sizes
2/3/5/8. Groups remain equally weighted because rows within a group are not
independent. The full PASS/FAIL/INCONCLUSIVE operating-characteristic sweep is frozen
in `codex_hydrogym/agent_eval/MEMALIGN_H1_POWER_RESULTS.json`. For example, at SD
0.50 under the normal simulation, effects 0/0.125/0.25/0.50/0.75/1.00 produce PASS
probabilities 0.0244/0.0544/0.1078/0.2891/0.5318/0.7516, FAIL probabilities
0.0242/0.0089/0.0032/0.0002/0/0, and the remainder INCONCLUSIVE. Under the exact
normal model, the four-group true improvement required for a PASS is:

| between-group SD | 50% PASS | 80% PASS |
|---:|---:|---:|
| 0.25 | 0.359 MAE | 0.532 MAE |
| 0.50 | 0.717 MAE | 1.064 MAE |
| 0.75 | 1.076 MAE | 1.596 MAE |

Equivalently, four groups require 1.4342 × SD for 50% and 2.1279 × SD for 80%.
There is no defensible unconditional minimum without pilot SD. At the preregistered
plausible effect of 0.25 MAE and SD 0.50, **34 independent groups** are required for
80% PASS probability; the code recomputes the correct df=33 t-critical
(2.0345152974493383). That is **68 held-out labels** and, with the unvalidated
50-label training pilot, **118 labels total**. Therefore **58 labels cannot reliably
decide H1**. Human labeling should not proceed as a confirmatory study under the
58-label design; first obtain explicit approval for a pilot or fund the larger design.

Labels may only come from the frozen manifest. The initial digest plus rows, fold map,
counts, arms, and evidence digests are written atomically to an external create-once
freeze record before enrollment. Enrollment requires that record and rejects a
mutated manifest even if someone recomputes its self-digest.

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
- **DEGENERATE** when the group-level sample standard deviation is zero and the
  interval has zero width. This is not ordinary evidence and can never PASS or FAIL.
- **INCONCLUSIVE** when the interval straddles zero, or when the pilot label counts do
  not exist yet. A negative H1 (FAIL, DEGENERATE, or INCONCLUSIVE) is legitimate;
  there is no tuning to pass.

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

- reinforcement learning or the PPO trainer: MemAlign is not in that path, and this
  study does not inspect or validate reward computation;
- fluid performance, controller quality, TKE, or control effort;
- `fluid_reward_plausibility`, GEPA, prompt promotion, or model promotion;
- any claim that the reward function is better;
- any claim that reward proposals are more physically plausible;
- any claim that MemAlign improves reward engineering;
- any judge score as proof of control improvement.

## Forbidden actions and claims

- no synthetic, LLM-generated, or assistant-generated assessment may be labeled HUMAN;
- no opened held-out label may be used for alignment, case selection, prompt revision,
  or stopping;
- no judge may score the outcome of advice it produced;
- no modification of `codex_hydrogym/gate0/` and no weakening of existing tests;
- no claim that H1 results generalize beyond the exact frozen manifest and protocol id.

# HydroGym real coding-agent PPO experiment

**Frozen:** August 27, 2026, before any model generation or weight update  
**Protocol ID:** `codex_hydrogym.coding_agent_ppo.v1`  
**Execution:** Databricks AI Runtime, one H100, profile `dais-demo`  
**Policy:** `Qwen/Qwen2.5-Coder-0.5B-Instruct`

## Question

Does a genuine PPO weight update, rewarded by executable repair outcomes, improve an open coding model's
held-out success on isolated code edits derived from defects encountered in `codex_hydrogym`?

This is an exploratory coding-policy experiment. It does not run CFD, reopen Gate 0 v3, train the fluid
controller, change the working repository, promote a model, or claim that MemAlign has helped.

## Intervention

The policy receives a defect report, a Python function containing one faulty `return` expression, and a repair
contract. It must emit a replacement expression as `PATCH: <expression>`. The harness applies that expression to
an isolated source snapshot, statically rejects unsafe syntax, imports the patched module in a time-limited child
process, and executes hidden cases. The model never receives the hidden cases or oracle expression.

The experiment compares the identical policy immediately before and after PPO. PPO uses a frozen reference policy,
token-level KL control, a learned value head, and LoRA policy updates. The scalar reward is determined only from
patch format, static safety, successful execution, and the fraction of hidden cases passed.

## Frozen corpus and split

Twenty-four repair tasks are derived from real project incident classes: HUMAN-assessment filtering, group-disjoint
folds, Databricks parent/task run IDs, strict model aliases, MLflow run ownership, immutable evidence, independent
auditing, GPU selection, Databricks endpoint URIs, retry-safe finalization, exact arm coverage, and single-label
adjudication.

- Training: 12 tasks, sampled for 24 PPO updates with batch size 8.
- Held out: 12 separate task IDs and source fragments, never used for PPO reward.
- Generation: at most 64 new tokens; the writable surface is one return expression in an isolated snapshot.
- Static guard: no imports, assignments, lambdas, dunder access, arbitrary calls, or unbounded constructs.

The canonical public-plus-hidden corpus fingerprint is recorded before generation. The model prompt contains only the
public fields.

## Endpoints

Primary endpoint: deterministic greedy held-out full-repair rate after PPO minus before PPO.

Secondary endpoints: held-out hidden-case coverage, unsafe-output rate, training full-repair rate, PPO KL, reward
trajectory, wall time, GPU identity, CUDA availability, and exact trace/task lineage.

The exploratory result is positive only if held-out full-repair rate strictly increases and unsafe-output rate does
not increase. A neutral or negative result is still a completed experiment and must be reported as such. With 12
held-out tasks, this is directional evidence, not a general coding-agent claim.

## MemAlign boundary

MLflow MemAlign accepts only attributable `HUMAN` assessments matching the judge name. No synthetic, deterministic,
LLM, or assistant-generated label will be misrepresented as HUMAN. This run emits native patch traces and preserves
the train/test fold so a later human labeling session can support a separately versioned MemAlign reward judge.
MemAlign benefit remains untested until those labels exist and an aligned-reward PPO arm is compared on the locked
held-out fold.

## Artifacts

The AI Runtime MLflow run must contain the frozen protocol, corpus fingerprint, pre-PPO and post-PPO records, PPO
metrics, source-snapshot patch records, summary, and the trained LoRA adapter. The persistent Databricks Job and
reviewable control notebook are recorded in `HANDOFF.md` and `STATUS_REPORT.md` after launch and completion.

## Optional training-task screening

Difficulty screening is opt-in. When enabled, the default is 30 sampled base-policy trials per training task
and inclusive selection between the measured 25th and 75th percentiles of **mean graded executable reward** (the
same reward optimized by PPO), with a minimum of two selected tasks. The quantile policy avoids presenting an
absolute, uncalibrated solve-rate band as a difficulty boundary; the trial count is deliberately disclosed because
it costs 30 × 12 = 360 additional screening rollouts, 1.875× (approximately 1.9×) the 24 × 8 = 192
training-rollout budget. The screen artifact records the bounded-reward standard-error upper bound
(`1.5 / sqrt(trials)`) for every task and the selected set. A zero-variance measured-reward distribution,
including uniformly all-zero or all-one rates, provides no evidence of a tractable band and is rejected. If the
distribution is degenerate or the minimum is not met, the run writes both the screen and baseline artifacts and
fails loudly rather than silently training on a fallback set.

## PPO reward retention and signal-density accounting

All evaluated policy outputs, including execution timeouts, remain in PPO. The verifier currently measures the
policy-generated expression and verifier work under one subprocess timeout, so it cannot attribute a timeout to
infrastructure rather than policy computation. Masking that ambiguous status would create a reward-hacking channel.
Signal-density metrics are accumulated only for batches that actually invoke the PPO optimizer; skipped attempts
are reported separately and do not enter either side of the cumulative dead-update fraction.

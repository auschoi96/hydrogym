# Progress review — August 29, 2026

**Audience:** the human owner and any agent picking this up cold.
**Predecessor:** `PROGRESS_REVIEW_2026-08-27.md`. Long-form detail lives in
`STATUS_REPORT.md` (83 KB) and `HANDOFF.md` (53 KB). This document is
deliberately short and is the honest current state, including the parts that
reflect badly on the orchestration.

---

## 1. The goal, restated

Show that an **agent harness can improve reinforcement learning**, using
hydrogym as the RL framework. Concretely: a coding agent authors or improves a
reward function / training setup for a fluid-control task, and we measure
whether the resulting policy is better against a metric fixed in advance that
the agent cannot touch.

That goal needs exactly five pieces. Here is what we actually have:

| # | Piece | Status |
|---|---|---|
| 1 | A hydrogym environment that runs | **HAVE IT.** JAX Kolmogorov steps on CPU, verified 2026-08-29 |
| 2 | A frozen scale-independent metric + baseline `C*` | Code exists, **UNWIRED**, `PREREGISTERED_ARTIFACT_PATH = None`, no `C*` ever measured |
| 3 | A coding agent that authors a fluid reward | **NEVER BUILT** |
| 4 | A PPO run that uses that reward | **NEVER RUN** |
| 5 | A comparison of (3)+(4) against the frozen metric | **NEVER DONE** |

**Eleven PRs were opened. None of them is piece 3.** Piece 1 was available the
entire time and nobody used it.

---

## 2. What has actually executed

Three things ran for real. This is the whole list.

### 2.1 Coding-agent PPO on one H100 — COMPLETED, NEGATIVE RESULT

Independently verified via the Databricks CLI on 2026-08-29:

- Job `683906346871429` "codex_hydrogym real coding-agent PPO v1"
- Run `622161538716123`, `TERMINATED / SUCCESS`, 1,845,896 ms (30.8 min)
- MLflow run `5517069a75334139ba8a7b8e27417828`
- 24 real PPO/LoRA updates over 192 rollouts

| Metric | Base | After PPO |
|---|---|---|
| Held-out exact repairs | 1/12 | **0/12** |
| Hidden cases | 15/36 | **7/36** |
| Unsafe outputs | 0 | 0 |

PPO made the agent **worse**. `HANDOFF.md:19` already said this plainly:
"It proves execution plumbing, not RL or MemAlign benefit."

**Two caveats that matter more than the numbers:**

1. **This is not the experiment the goal asks for.** The goal is *use an agent
   harness to improve RL*. This run *used RL to train a coding agent* on
   code-repair tasks. It is a different question, on a non-fluid task. The
   substitution was never authorized and is the clearest single piece of
   evidence of scope drift.
2. **RESOLVED 2026-08-29 — the run is a VALID FINDING, but the headline number
   is statistically vacuous.** See §2.1a.

### 2.1a Forensic verdict on the PPO negative result

The worry was that the run executed on a defective harness (a fatal wiring
bug, a screening gate that can hand PPO an all-zero-reward pool, a timeout
path that masked slow policy outputs out of training) and was therefore an
artifact rather than a finding. **That worry is refuted.** The defective code
did not exist when the run executed.

Verified directly with `git log --all -S`, not taken on a worker's word:

| Symbol | First enters git | vs. run end (2026-08-27 21:14 UTC) |
|---|---|---|
| `_INFRASTRUCTURE_FAILURE_STATUSES` | `6578ec6` 2026-08-29 17:56 | +2 days |
| `select_tasks_by_solve_rate_quantile` | `df738cb` 2026-08-29 18:22 | +2 days |
| `masked_ppo_step` | `df738cb` 2026-08-29 18:22 | +2 days |

All three live only on unmerged `polly/ppo-signal-density*` branches. They
cannot have affected a run that finished two days earlier. Consequence: the
PR #7 defects and their fixes are **forward-looking only** — they gate the
*next* code-repair run and say nothing about this one.

What the run actually executed: untracked working-tree code on top of commit
`d708dd0` (which contains **no** `codex_hydrogym/` directory at all — verified
by `git ls-tree`). `dirty: true, diff_status: clean` in the launch
`git_state.json` therefore means untracked files, not a modified tracked tree.
Three independent signatures pin the executed source to `6b256b2`-era code:
the run's logged `protocol_sha256` matches
`git show 6b256b2:.../CODING_AGENT_PPO_PROTOCOL.md`; the run logs
`coding_ppo.trl_running_moments_tensor_compat`, a param only that era emits;
and the fixed code's screening params are absent from the run. The byte-exact
tree is unrecoverable — it was never committed. That is a real gap in the
record, and the reason it is not fatal is that the defects' source provably
did not exist anywhere in history until two days later.

From the run's own telemetry: **0** rollouts with status `execution_timeout`,
**0/24** dead updates, reward spread `−1.25 … +1.25` present in every batch,
`objective_kl` 0.003 → 2.84, and full repairs occurring in updates 3, 4 and 7.
PPO had real signal. The "nothing to learn from" scenario did not occur.

**But the headline number must be retired.** Exact binomial arithmetic,
reproduced locally against `scipy` at n=12, p₀=1/12:

```
P(X=0 | p0)   = 0.3520   <-- the observed "0/12 regression"
P(X>=2 | p0)  = 0.2640   <-- even a DOUBLING is 26% likely under the null
P(X>=3 | p0)  = 0.0720
P(X>=4 | p0)  = 0.0138
Fisher [[1,11],[0,12]] two-sided p = 1.0
```

The observed "regression" to 0/12 is the **single most likely outcome of no
change at all** (35%). The only exact one-sided levels available are α≈0.072
(reject at ≥3/12) or α≈0.014 (reject at ≥4/12). Power to detect a genuine
tripling of the repair rate is 0.609 at the looser level and 0.351 at the
stricter one. **The exact-repair endpoint cannot answer its own question in
either direction.** A 2/12 or 3/12 "improvement" would have been equally
uncitable.

Corpus size needed before the code-repair endpoint can carry a claim: **31
held-out tasks** for 80% power to detect a tripling, one-sided α=0.05, when
the 1/12 baseline is treated as fixed (reject at ≥6/31, power 0.824). If both
arms are estimated from the same small corpus, roughly double that. Either
way the current corpus of 12 is short by 2.5–5×.

The degradation signal that *is* real lives in the secondary metric: hidden
cases 15/36 → 7/36, 4 tasks degraded / 0 improved / 8 tied. Paired sign test
one-sided p = 0.0625; unpaired Fisher one-sided p = 0.0360, two-sided
p = 0.0721 — all three reproduced locally. Nominally significant, unadjusted
for two-metric multiplicity, so: **modest real evidence of harm, not a clean
result.** Case-level McNemar is not identifiable because `case_records` carry
no case IDs. The policy's post-update failure mode is mostly syntactic
collapse (5 train rollouts ended `syntax_error`), which is itself consistent
with real, harmful policy movement rather than a no-op.

**How to cite this run, precisely:** PPO on a 0.5B model over 192 rollouts
moved the policy and moved it the wrong way on partial credit. It is not
evidence that RL cannot help a coding agent, and the 1/12 → 0/12 figure must
not be quoted as evidence of harm.

### 2.2 Fixed-controller fluid replication — COMPLETED, POSITIVE

Ten new seeds, 120 trajectories, aggregate base-condition TKE reduced
**41.258989%**, every numerical / causal / window-stability / point-convergence
/ equivalence predicate passed. Independently reproduced by a standard-library-
only audit (120 traces, 240 windows, 840 gate evaluations) with **zero CFD**.
This is a hand-designed gain-2 signed forced-mode feedback controller. **No RL.**

### 2.3 Gate 0 v3 — COMPLETED, FAILED ITS OWN GATE

360 trajectories, 720 windows. Failed exactly one locked conjunctive predicate:
temporal feedback-vs-zero 90% equivalence interval lower endpoint
`-0.020255923821008402` against a frozen boundary of `-0.02` — missing by
`0.000255923821008402`.

**Nobody widened the boundary.** Given how much of this project has been
catching itself in self-deception, honoring a preregistered gate you miss by
two ten-thousandths is the most credible thing in the repository.

---

## 3. The ratio problem

| Classification | PRs | Lines added |
|---|---|---|
| Measurement apparatus | 9 | 9,732 |
| Infrastructure (#4, #9) | 2 | 564 |
| **Actual experiment** | **0** | **0** |

~10,300 lines. Zero experiments. **Eleven PRs open, none merged.**

### Where the apparatus was genuinely load-bearing

Do not overcorrect and throw all of it out. Three pieces earned their place:

- **UC OTel tracing (PR #9)** — the one thing in this project independently
  reproduced end-to-end. An adversarial auditor re-ran the probe three times
  from its own worktree into its own experiment and table prefix, on mlflow
  3.15.1 *and* 3.11.1, reading every row back from Unity Catalog. Zero blocking
  findings. It also **disproved the repo's own `mlflow >= 3.14` pin** — 3.11.1
  creates all four OTel tables.
- **The frozen-metric guard (PR #10)** — the human's own hard requirement. The
  audit found the deployed PPO workload *bypassed the guard entirely*. Without
  the audit we would have believed a metric the agent could edit was protected.
- **Null-control calibration (PR #11)** — cleared with zero blocking, all eight
  audit blocks re-derived exactly, and it produced the single most reusable
  finding in the repo (see §4).

### Where cross-review earned its keep

Five audits caught four real self-deceptions that would have made any result
fake: the fatal `AttributeError` wiring bug; the all-zero screening pool; a
vacuous calibration assertion; and a false version pin. Cross-vendor review is
not the problem.

**The problem is the ratio: five audits, zero experiments.** Rigor was treated
as a deliverable to be built to completion before running anything, instead of
as a constraint on an experiment that should have been run first. The minimum
honest version of piece 2 is one file and one baseline run — not 10,300 lines.

---

## 4. Reusable findings

- **A vacuous CI gate.** A test asserting "the confidence interval's upper bound
  is >= X" when the true rate is *below* X is a **luck gate, not a calibration
  gate**: its pass probability *peaks at the smallest allowed sample size* and
  decays to zero as evidence accumulates (0.74 at n=200, 0.014 at n=1000 for a
  true rate of 0.025). It gates on ignorance. Asked at what n it becomes
  reliable for a true rate of 0.032, the answer is **never**. Grep this
  repository for the same pattern.
- **Exact equality is not a degeneracy check.** PR #7's screening rejected only
  `ordered[0] == ordered[-1]`. Any "does this distribution have variation" guard
  must compare spread against **the estimator's own uncertainty**, and must
  check the **selected** subset, not just the measured pool.
- **Firedrake was never the blocker on fluid RL.** It is not installed and is
  not pip-installable, and its absence was treated all project long as blocking
  fluid work. It is not: `hydrogym.jax.envs.kolmogorov` steps on CPU
  (`reward -2.0585186`, obs shape `(16,)`, `[CpuDevice(id=0)]`). No fluid PPO
  ran because nobody ran it.
- **`examples/jax/getting_started/3_ppo/run_ppo.py` is a smoke config, not a
  training config** — `TOTAL_TIMESTEPS=100` with `NUM_ENVS=4, NUM_STEPS=40`,
  fewer total timesteps than a single rollout. Never used for a real run.
- **Verify the pin, not just the guard.** When a digest-pinned file legitimately
  changes, its pin must change too — byte-for-byte indistinguishable from
  tampering unless someone independently recomputes the hash. Any task editing a
  pinned file must print `shasum` and the pin constant side by side.

---

## 5. Operational notes for other agents

- **`databricks` CLI fails from the repo root** on a bundle-host mismatch. Run
  it from `/tmp` with `--profile dais-demo`.
- **Lint gates must be scoped to touched files.** The repo has ~48 pre-existing
  repo-wide `ruff` errors including notebook `await` syntax in untouched files.
  It has never been repo-wide clean; every prior "All checks passed" was
  file-scoped.
- **Branch sprawl.** ~30 worktrees, 11 unmerged branches. A file absent from
  HEAD may exist elsewhere: `codex_hydrogym/memalign_h1/` has **zero files** at
  `5fc9bd5` and exists only on `polly/memalign-h1-round2` (`3cf1d05`). Always
  `git ls-tree -r --name-only <sha>` before citing a path.
- **Test counts:** collected *cases* are ground truth
  (`pytest --collect-only -q`). Never use `grep -c 'def test_'` as an oracle —
  parametrized tests expand one function into many cases.
- **Worker roster is thin.** Only two usable vendors. `claude_code`,
  `opencode`, `hermes` have no usable model provider; `agy` is not installed;
  `codex` failed to boot three times. **All `cursor` sessions share one runner
  and cannot run concurrently.** Long-running compute tasks sent to `pi` have
  died with "Pi process ended without response" four times — `pi` is reliable
  for read/analyze work and unreliable for heavy execution.
- **Do not invent model endpoint names.** Two "unavailable model" conclusions
  were actually guessed endpoint names returning 404. A 429 means the endpoint
  resolves and is rate-limited (recoverable); a 404 means the name is wrong.

---

### The worker environment contract (read this before you run any script)

Six sub-agent sessions died with "process ended without response" and were
misattributed to heavy compute or a bad model. The real cause is
environmental, and it is now verified with shell commands:

1. **`/tmp` is not writable.** A worker's `sys_os_write` to `/tmp/foo.py`
   returns `Access to '/private/tmp/foo.py' is blocked: path is outside the
   environment root '/Users/austin.choi/PycharmProjects2/hydrogym'`. Workers
   silently fall back to a shell heredoc into `/tmp`, which sets up failure 2.
2. **`hydrogym` is not pip-installed.** `pip show hydrogym` reports
   `Package(s) not found`. The package resolves *only* through the implicit
   `sys.path[0]` when cwd is the repo root. A script sitting in `/tmp` dies on
   `ModuleNotFoundError: No module named 'hydrogym'`. This is exactly what
   killed the first two fluid-PPO attempts, before either ran a single step.
3. **Bare `python` is the wrong interpreter.** On PATH it resolves to
   `/Users/austin.choi/PycharmProjects2/omniagent/.venv/bin/python`, then
   `/opt/anaconda3/bin/python`. Neither has `jax` or `hydrogym`.

Fix applied: `.scratch/` now exists at the repo root and is gitignored, giving
agents a legal in-root scratch location. The contract for every command is:

```
cd /Users/austin.choi/PycharmProjects2/hydrogym && ./.venv/bin/python .scratch/<script>.py
```

Verified working env construction (CPU only, `[CpuDevice(id=0)]`):

```python
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow
env = KolmogorovFlow(
    env_config={'grid_size': (16, 16), 'obs_size': 4, 'max_episode_steps': 2},
    flow_config={'grid_size': (16, 16), 'obs_size': 4},
)
# reset -> obs shape (16,); one step -> reward about -2.0585186
```

The general lesson is worth more than the fix: a worker that dies without a
report is usually blocked on the environment, not on the task. Mine its
conversation history for the last tool error before assuming the model or the
compute budget was at fault.

---

## 6. Open decisions for the human

1. **Judge name** — `critic_quality` vs `fluid_reward_plausibility`.
2. **Protocol rule** — two independent audits now argue FOR adopting the
   group-clustered interval rule that `AGENT_REVISION_PROTOCOL.md:45`
   preregisters but the executed code replaces with a bare sign test.
3. **Human labels** — do **not** authorize 118 expert labels yet. That figure is
   conditional on invented priors (0.25 MAE, an ungrounded SD grid, and zero
   existing human labels). Run a 2–3 group variance pilot first. The
   unconditional fact: 4 groups at df=3 need >= 1.4342 x SD for coin-flip
   detection.
4. **MemAlign cannot run at all** — zero eligible human labels exist.

---

## 7. Recommended next step

One thing, in this order, and nothing else until it is done:

1. Determine what one honest fluid PPO run costs (in flight: `fluid-ppo-smoke2`).
2. Run it. Preregister the scale-independent metric and the decision rule
   **before** training and never touch them afterwards.
3. Only then build piece 3 — the coding agent that authors the reward.

A negative or null fluid result is a completely acceptable deliverable and is
worth more than another audit. Stop building apparatus.

### What §2.1a changes about priority

The code-repair PPO lane is now **doubly deprioritized**, on two independent
grounds:

- It answers the inverse question (RL trains a coding agent) rather than the
  goal (an agent harness improves RL), on a non-fluid task.
- Its evaluation endpoint is unpowered at n=12 and needs a **31–60 task
  held-out corpus** before it can carry any claim in either direction. That is
  a corpus-construction project, not a code fix.

So the PR #7 screening/masking fixes are correct but **not on the critical
path** — they harden a lane whose measurement instrument is the binding
constraint. Land them if cheap; do not queue more work behind them.

The fluid lane has no such problem: `hydrogym.jax.envs.kolmogorov` steps on
CPU today, and a continuous control metric (normalized TKE reduction at
matched actuation cost) does not suffer the discrete floor effect that makes
1/12 unmeasurable.

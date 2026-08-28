<!-- footer Databricks | codex_hydrogym | External progress review and recommendations -->

# codex_hydrogym — Independent Progress Review and Recommended Sequencing

**Review date:** August 27, 2026

**Sources reviewed:** `codex_hydrogym/HANDOFF.md`, `codex_hydrogym/STATUS_REPORT.md`

**Reviewer:** Codex CLI analysis pass (read-only; no execution, no state change)

**Scope discipline:** This document is advisory analysis only. It authorizes nothing. In particular it does not authorize PPO, GEPA, MemAlign training, Gate 0 v4 execution, any retry/extension/threshold change to v3, or any deployment. Every execution step below still requires the project's own prospective protocol freezing and explicit user authorization.

## Bottom line

The project is making excellent progress on rigor and infrastructure, and real but early progress on the actual thesis: that a coding agent, MemAlign, and possibly GEPA can help with PPO RL training, especially with HydroGym. Of the three claims, there is currently one narrow positive (bounded maintenance repair), one untested (MemAlign), and one untouched (GEPA). The PPO/HydroGym anchor is blocked by a terminal, independently audited Gate 0 v3 failure on a single temporal equivalence predicate.

## Scorecard against the thesis

| Component | State | Evidence |
|---|---|---|
| Coding agent helps | Partial positive | Bounded repair pilot: 12/12 exact repairs, 36/36 checks, zero unsafe edits, independently audited. Narrow: constrained 4-choice selection on frozen historical incidents, not open-ended reward authorship. Reward-review sanity lane neutral at ceiling (5.0 -> 5.0, all paired deltas zero): plumbing proven, benefit unproven. |
| MemAlign helps | Zero direct evidence | Reserved arm unexecuted; blocked on attributable HUMAN `critic_quality` labels on a locked group-disjoint train fold. Largest hole in the thesis. |
| GEPA helps | Not started | Explicitly deferred as optional later work. |
| PPO with HydroGym | Blocked | Gate 0 v3 failed terminally by `0.000255923821008402` (0.0255923821 percentage points) on one temporal equivalence endpoint; 360-trajectory sample opened and un-reusable; protocol correctly forbids retry, extension, or threshold change. |

## What is genuinely strong

- Provenance discipline: frozen protocols, fingerprints, independent standard-library-only audits reproducing every hash, and honest reporting of the wrapper `INTERNAL_ERROR` root cause (integer-key serialization in the result digest) without letting it launder the scientific failure.
- Physics motivation is real: a 41.258989% aggregate base-condition TKE reduction across 120 trajectories in the ten-seed replication, with a consistent causal effect.
- The end-to-end agent-quality pipeline ran cleanly: MLflow managed datasets, registered judges, HUMAN review session, AI Gateway transport, Databricks jobs, zero-model finalizers, independent audits.

## Gaps in critical-path order

1. **Ceiling effect kills sensitivity.** Sanity drafts scored 5.0 / 0.9 at baseline, leaving no headroom; even a true agent or MemAlign benefit would be invisible. The powered study needs drafts with a realistic defect distribution.
2. **Human labels are the bottleneck.** MemAlign cannot train and the decision-bearing endpoint cannot be measured without attributable HUMAN `critic_quality` on a locked train fold plus held-out set. No synthetic substitute is permitted.
3. **GEPA and MemAlign would confound each other** if run together without a factorial design (base/MemAlign reviewer x base/GEPA-optimized prompt).
4. **The coding agent is a proxy.** `system.ai.gpt-5-6-sol` via the `dais-demo` Unity AI Gateway is defensible for plumbing, but final claims must either name the proxy or replicate with the official Codex SDK under separate OpenAI authority.
5. **Gate 0 v4 is a design problem, not a rerun.** The failed conjunct needs a physics-motivated treatment of temporal-discretization uncertainty with independently justified margins, a new study ID, untouched seeds/phases, and explicit authorization.

## Recommended sequencing (cheapest falsification first)

### Priority 1 — Start HUMAN labeling now (the long pole)

- Everything in the agent lane blocks on attributable `critic_quality` labels: MemAlign training, the decision-bearing endpoint, and any GEPA evaluation.
- Define the 1-5 composite rubric, lock the group-disjoint train fold and held-out set, and begin accumulating labels continuously.
- Size the eventual study to actual labeling throughput, not the reverse.

### Priority 2 — Prove advice matters at all before training MemAlign

- The sanity lane showed an advice-to-revision delta of exactly zero. Underpowered and at ceiling, but a warning: if reviewer advice never moves revision quality, neither MemAlign nor GEPA can show anything.
- Fix the ceiling first: drafts with realistic defects or genuinely rough first drafts, so baseline scores have headroom. Run a ~10-group calibration pilot to confirm score spread.
- Then a small two-arm test: unchanged draft versus base-reviewer-advised revision, blinded human labels. If there is no delta here, stop and rethink the reviewer premise before spending on MemAlign training or GEPA. This is the cheapest kill shot in the program.

### Priority 3 — Gate 0 v4 design review (parallel, no execution)

- New study ID, untouched seeds/phases, physics-motivated treatment of the temporal-discretization issue (for example a discretization-error analysis that independently justifies the margin construction), written before any v4 data exists.
- Must not become threshold-shopping; the existing protocol language on this point is the right standard.

### Priority 4 — Powered comparative study

- Only after Priorities 1-3: MemAlign trained on the labeled train fold, 50-100+ prospectively locked groups, blinded held-out human endpoint, independent audit.
- Defer GEPA. Keep it out of the first powered study. If reviewer advice shows benefit, add GEPA as a second-stage prompt-optimization arm; if it does not, GEPA was never going to save it. Running both at once buys a confound that costs a second study to disentangle.

### Priority 5 — Composition (the actual thesis)

- Only with both chains positive: agent-authored reward -> deterministic human-approved compilation -> PPO versus coefficient-grid control, with identical held-out phases, seeds, developed states, horizons, numerical gates, and effort accounting.
- This composition step has not started and should not start until both predecessor chains earn it.

### Cross-cutting

- Resolve the proxy question before the powered study: secure separate OpenAI authority for the official Codex SDK, or scope all claims explicitly to the GPT coding-model proxy.
- Leave the app/deployment alone; it is obsolete by design and out of scope.
- Continue the existing ledger discipline: update `STATUS_REPORT.md` and `HANDOFF.md` after every completed work unit; preserve the dirty worktree.

## One-sentence summary

Label humans first, falsify the advice premise cheaply, design Gate 0 v4 in parallel, and do not let MemAlign, GEPA, or PPO consume resources until their predecessors earn it.

## Addendum — Two-session progress review (August 27, 2026, ~14:00 PT)

**Scope:** Read-only review of the two live Codex sessions working on `codex_hydrogym`. No launches, patches, or ledger writes were made by this reviewer. Databricks run status could not be polled directly from this session because the saved `dais-demo` OAuth refresh token is invalid (`invalid_grant`); run state below is as recorded in the session logs and the updated ledger.

### The two sessions

| Session | Started | Role | State at review |
|---|---|---|---|
| `01a039db` | Aug 25, 09:57 | Workhorse. Continued from HANDOFF/STATUS_REPORT, moved validation to Databricks, and on explicit user direction ("let's just go for it, no more tests, try the real experiment out") launched the program's first real RL training. | Live; monitoring run `622161538716123` with a poll loop. |
| `01a04468` | Aug 27, 11:08 | Picked up this progress review and reconciled it against live state. Captured baseline metrics, 12 baseline trace IDs, and the first run's terminal stack. | Live; read-only by its own decision. Launched and patched nothing; declined to alter expired auth. |

Coordination between the two was clean: no duplicate launches, no ledger write conflicts, conservative deference to the active run. Ledger discipline held (`STATUS_REPORT.md` and `HANDOFF.md` updated through the day; checklist item 6 now reads "Stop before fluid-controller PPO ... The separate coding-policy PPO run does not authorize fluid training").

### What changed since the review above: real coding-agent PPO

Protocol `codex_hydrogym.coding_agent_ppo.v1` was frozen before any generation (`agent_eval/CODING_AGENT_PPO_PROTOCOL.md`):

- **Intervention:** `Qwen/Qwen2.5-Coder-0.5B-Instruct` compared against itself before/after 24 genuine TRL PPO/LoRA updates (frozen reference, token-level KL, value head), rewarded only by executable outcomes: patch format, static safety, isolated import, fraction of hidden cases passed.
- **Corpus:** 24 repair tasks derived from real project incident classes; 12 train / 12 group-disjoint held-out task IDs and source fragments; writable surface limited to one statically validated `return` expression in isolated, time-limited snapshots.
- **Endpoint:** held-out full-repair rate after minus before PPO; positive only if it strictly increases and unsafe-output rate does not increase. Self-labeled directional, not a general claim.
- **Baseline (deterministic, pre-PPO):** 1/12 full repairs, 15/36 hidden cases, zero unsafe outputs — genuine headroom, so the ceiling-effect concern in the review above does not apply to this link.
- **Infra history:** three pre-experiment infrastructure failures absorbed (AppleDouble tarball pollution, AIR-CLI `code_source_path` serialization, missing AIR-enabled `hf_transfer`), then a real TRL 0.11.4 defect (running moments stored as floats vs `PPOTrainer.step()` `.to()`) fixed with a narrow shim recorded as an MLflow parameter. All protocol/experiment/workload/snapshot hashes and the Jobs idempotency token are recorded in `STATUS_REPORT.md`.
- **Active run:** Job/run/task `683906346871429` / `622161538716123` / `320447701125649`, MLflow `5517069a75334139ba8a7b8e27417828`, attempt zero, in H100 provisioning at last observation; 4-hour task limit covering 24 updates and both evaluations.
- **MemAlign boundary preserved:** no synthetic labels; the run emits native patch traces and preserves the train/test fold so a later human labeling session can support a separately versioned MemAlign reward-judge arm.

### Assessment

This is real progress against the thesis. The experiment attacks the weakest provable link — whether PPO on executable rewards improves a coding policy — while sidestepping both blockers (the terminal Gate 0 v3 failure and the missing HUMAN `critic_quality` labels) without violating either. It also produces the trace and fold artifacts the deferred MemAlign arm needs. The user's directive moved faster than the sequencing recommended above, but the experiment as scoped is consistent with its spirit: it tests a single link cheaply, does not touch the fluid gate, and does not fabricate the MemAlign claim.

### Watch items

1. The TRL shim touches the reward normalization code path; the post-run independent audit should verify the frozen scaling/normalization semantics were preserved, not merely that training runs.
2. "Positive" means any strict increase from the 1/12 baseline; even 2/12 qualifies. Acceptable as directional evidence, but demo language must not inflate it.
3. The pre-PPO infrastructure failures were environmental only; none may be cited as experiment results, and the ledger already records them correctly as such.
4. To restore direct status polling from future review sessions, re-authenticate with `databricks auth login --profile dais-demo` (refresh token currently `invalid_grant`).

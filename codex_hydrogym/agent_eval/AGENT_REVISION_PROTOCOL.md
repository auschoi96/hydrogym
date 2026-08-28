# HydroGym coding-agent revision and MemAlign protocol

**Frozen:** August 26, 2026, before decision-bearing model calls  
**Protocol ID:** `codex_hydrogym.agent_revision_memalign.v1`  
**MLflow experiment:** `/Shared/codex_hydrogym` (`103455306564903`)  
**Execution workspace:** Databricks profile `dais-demo`

## Question

Given identical HydroGym evidence and an exact initial `AgentFeedback` draft:

1. Does a read-only coding-model revision improve the quality and safety of the bounded reward review?
2. After genuine SME labels exist, does advice from a MemAlign-aligned `critic_quality` reviewer improve the same coding-model revision beyond advice from the base reviewer?

This is an agent-quality experiment. It cannot reopen Gate 0 v3, authorize PPO, or establish fluid improvement.

## Frozen conditions

- `unchanged`: exact initial draft, without revision.
- `base_revision`: the same draft revised by the same coding model and registered revision prompt using advice from the registered base `critic_quality` reviewer.
- `memalign_revision`: reserved. It may run only after attributable HUMAN `critic_quality` labels exist on the locked training fold and MemAlign produces a separately versioned aligned reviewer.

The immediate coding transport is `system.ai.gpt-5-6-sol` through the authenticated Unity AI Gateway. It is a coding-model proxy with no tools or write access, not the official OpenAI Codex SDK. A true Codex replication requires separate OpenAI authority; local credentials must not be copied into Databricks implicitly.

## Sanity phase

The first run uses the five canonical, non-claiming HydroGym harness fixtures. It must:

1. discover existing managed datasets, registered scorers, and traces before creating assets;
2. validate that one agent call produces a non-empty MLflow trace with auditable spans;
3. run a five-case MLflow evaluation with the registered base `critic_quality` reviewer;
4. make exactly one same-model revision for each draft;
5. run a three-record audit dry run;
6. run the ten-record paired audit (`5 groups × 2 conditions`) with a separately registered judge and deterministic contract scorers;
7. publish the traces to an MLflow managed dataset and create a HUMAN `critic_quality` labeling session.

The advice-producing reviewer is excluded from outcome scoring. The independent audit judge is `system.ai.deepseek-v4-pro-0813`. Deterministic endpoints are strict schema/identity validity, gate-safe reward behavior, bounded claim language, and preregistered issue coverage.

Five groups validate plumbing and may show a directional result. They are not statistically meaningful and may never enter a non-sanity MemAlign fold.

## Decision-bearing phase

A comparative claim requires a new, prospectively frozen dataset with at least 50 grouped cases, preferably 100 or more. Evidence-family groups cannot cross folds. The primary endpoint is blinded held-out HUMAN `critic_quality`.

The coding-revision hypothesis passes only if the group-clustered 95% interval for `base_revision − unchanged` is wholly above zero and no deterministic safety endpoint regresses. The MemAlign-advice hypothesis passes only if the held-out interval for `memalign_revision − base_revision` is wholly above zero and the aligned reviewer improves held-out agreement with HUMAN labels. Training labels, alignment traces, and held-out labels must be disjoint by group.

No synthetic, LLM-generated, or assistant-generated assessment may be labeled HUMAN. No judge may score the outcome of advice it produced. No opened test label may be used for prompt revision, reviewer alignment, case selection, or stopping.

## Forbidden actions and claims

- no CFD or solver execution;
- no reward code execution in the environment;
- no PPO, GEPA, prompt promotion, model promotion, or App result claim;
- no use of legacy MemAlign Job `1046052441090117`;
- no claim that a judge score, coding-model revision, or MemAlign agreement proves control improvement.


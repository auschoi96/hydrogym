# codex_hydrogym implementation lane

Project label: `codex_hydrogym`

This directory is the manifest for the DAIS "Turbulence in the Loop" implementation.
All new demo resources, MLflow runs, Databricks resources, generated artifacts, and
test lanes use `codex_hydrogym` as their prefix or project tag.

## Work boundary

- Baseline commit: `d708dd0`
- Implementation branch: `codex_hydrogym/physics-and-harness`
- First gate: make the JAX Kolmogorov physics trustworthy on CPU before GPU work.
- Workspace policy: no Databricks workspace mutation until the user selects an
  explicit CLI profile.
- Unrelated user-owned paths such as `.isaac/` and `test/claude_hydrogym/` are not
  part of this lane and must remain untouched.

## Verification convention

Tests owned by this lane live under `test/codex_hydrogym/`. Every implementation
increment is required to show its focused tests passing before the next increment.

## MLflow controller lifecycle

A physics-valid PPO run logs a Models-from-Code pyfunc named
`codex_hydrogym_controller`. The model contains the deterministic clipped policy,
strict observation/action signature, exact dependency manifest, validated config,
physics-gate report, full checksummed restart checkpoint, and source-run
fingerprints. AI Runtime also requires an explicit three-level Unity Catalog model
name whose leaf begins with `codex_hydrogym`.

New versions receive only the `candidate` alias. The bundle job
`codex_hydrogym_promote_controller` can move `production` after it verifies that the
version belongs to the candidate MLflow run, both comparable runs pass every physics
gate, held-out mean TKE improves by the configured threshold, and control effort
stays within budget.

# Databricks notebook: ensemble replication

`ensemble_replication.py` is the reviewable Databricks source notebook for the exact frozen ten-seed Re=100 study. It is uploaded in profile `dais-demo` at:

`/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/ensemble_replication`

The notebook defaults to `action=review`, which executes no CFD. `action=preflight` verifies the wheel, protocol, all eight implementation-file hashes, transition records, Python 3.12, one H100, JAX GPU execution, and x64. `action=run` requires the exact confirmation token displayed in the notebook.

Scientific boundary: the user-directed platform switch was made without opening the one completed local base-condition artifact. That artifact remains quarantined and excluded. The full Databricks run is the sole analysis set. The later GPU amendment changes only the execution backend; seeds, phases, thresholds, precision, and analysis remain frozen. This is not PPO and cannot itself authorize PPO.

Execution uses Databricks AI Runtime on one `GPU_1xH100` because the H100 is expected to reduce wall-clock time for the JAX rollouts and supports float64. The AIR entry point fails before simulation unless it sees one H100, GPU-backed JAX, x64, the exact package pins, and all digest-bound artifacts.

Companion artifacts uploaded beside the notebook:

- `hydrogym-1.0.0-py3-none-any.whl`, SHA-256 `91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1`;
- `platform_transition.json`, artifact digest `deb256b550dd7d3d0fc88746db4ca7b0cbcdeb60f8b921d4fe19bb0466ad2e8a`;
- `execution_backend_amendment.json`, artifact digest `28747c56c53d2dd251dee8f17f49ef1c67c5d5b01d5fc9b3ea9d1b8f8c84e181`.

The full AIR workload is `codex_hydrogym/deploy/air/workload.gate0-h100.yaml`; its fail-closed entry point is `codex_hydrogym/deploy/air/gate0_replication_entrypoint.py`. The broad root bundle is intentionally not used because it contains legacy jobs outside this study's authorization boundary.

## Gate 0 v3 terminal independent audit

`re100_v3_terminal_audit.py` is the read-only Databricks source notebook for independently auditing the completed held-out Gate 0 v3 evidence. It is uploaded with profile `dais-demo` at:

`/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/re100_v3_terminal_audit`

Notebook object: `4282786714316185`. Persistent CPU-serverless Job: `372974263929719`.

It uses only the Python standard library and imports neither HydroGym nor the production analyzer. It runs no CFD or RL. Its scope is the exact seven-file terminal namespace, 360 traces, 720 windows, 2,520 numerical gates, all controller/history/derangement identities, all 60 condition-level primary gates, and an exact recomputation of every result statistic and predicate. CPU serverless is deliberate: GPU compute would not accelerate this read/hash/scalar-statistics workload.

Successful run/task `933651969763400` / `1023830567796956` completed on attempt zero. Local and exported workspace notebook source both have SHA-256 `8bb4d76a7bb52bf961e53169c30f9386e530ea77e395a49e2d5d3c71547e9d19`. It reproduced exactly one false predicate, `temporal_all_effect_equivalence_intervals_inside_margin`, and classified the primary wrapper failure as an integer-to-string JSON key canonicalization defect in 15 nested seed-cluster maps. The raw result bytes and typed pre-serialization digest are valid; the ordinary post-JSON canonical digest is not, which is why the producer stopped before writing its AIR summary.

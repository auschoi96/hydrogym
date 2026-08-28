# codex_hydrogym continuation handoff

**Prepared:** August 27, 2026  
**Repository:** `/Users/austin.choi/PycharmProjects2/hydrogym`  
**Primary status document:** `codex_hydrogym/STATUS_REPORT.md`  
**Original Codex session referenced by the user:** `01a02f5d-8563-7db0-b19c-88edc7aca608`

## Read this first

This project is promising but not RL-ready. The strongest honest conclusion is:

- a hand-designed gain-2 signed forced-mode feedback controller has a large and highly consistent causal effect at Re=100;
- the frozen ten-new-seed Databricks replication completed all 120 trajectories, reduced aggregate base-condition TKE by `41.258989%`, and passed every numerical, causal, window-stability, point-convergence, and uncertainty-aware equivalence predicate;
- an independent, standard-library-only Databricks audit reproduced all hashes, 120 traces, 240 windows, 840 numerical-gate evaluations, and the positive screen without running CFD or importing the production analyzer;
- the separately frozen held-out Gate 0 v3 then completed all 360 trajectories and 720 windows, but failed exactly one conjunctive predicate: the temporal feedback-versus-zero 90% equivalence interval lower endpoint was `-0.020255923821008402`, outside the frozen `-0.02` boundary by `0.000255923821008402` (`0.0255923821` percentage points);
- every base causal/numerical predicate, all point-convergence predicates, all other temporal equivalence intervals, and every spatial predicate passed, but one false predicate is still a terminal v3 failure;
- the outer AI Runtime task ended `FAILED`/`INTERNAL_ERROR` after persisting the immutable result and before writing `air_run_summary.json`; this wrapper failure cannot turn the scientific failure into a pass;
- no retry, threshold relaxation, sample extension, fluid-controller PPO, MemAlign training, GEPA, model promotion, or App deployment is authorized by this evidence;
- the separately authorized coding-policy experiment completed 24 real PPO/LoRA updates on one H100, but held-out exact repair fell `1/12 -> 0/12` and hidden-case coverage fell `15/36 -> 7/36`, with zero unsafe outputs in both conditions. It proves execution plumbing, not RL or MemAlign benefit, and cannot change the fluid gate.

## Real coding-agent PPO completed — negative result

The user explicitly authorized a real coding-policy PPO experiment on August 27, 2026. This is separate from the failed fluid gate and does not execute CFD or train the HydroGym controller. Protocol `codex_hydrogym.coding_agent_ppo.v1` is frozen in `codex_hydrogym/agent_eval/CODING_AGENT_PPO_PROTOCOL.md`. It compares the identical `Qwen/Qwen2.5-Coder-0.5B-Instruct` policy before and after 24 TRL PPO/LoRA updates, using executable hidden-case reward for one-expression edits applied only to isolated source snapshots. The corpus contains 12 training and 12 group-disjoint held-out task IDs derived from project incident classes. The directional endpoint is held-out full-repair-rate improvement without an unsafe-output increase.

- Review notebook: object `2854344201905376`, `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_agent_proof/coding_agent_ppo_real_experiment`.
- Direct review links: [notebook](https://fevm-austin-choi-omni-agent.cloud.databricks.com/editor/notebooks/2854344201905376?o=7474647489683936), [persistent Job](https://fevm-austin-choi-omni-agent.cloud.databricks.com/?o=7474647489683936#job/683906346871429), [successful run](https://fevm-austin-choi-omni-agent.cloud.databricks.com/?o=7474647489683936#job/683906346871429/run/622161538716123), and [MLflow result](https://fevm-austin-choi-omni-agent.cloud.databricks.com/ml/experiments/103455306564903/runs/5517069a75334139ba8a7b8e27417828?o=7474647489683936).
- Persistent AI Runtime Job: `683906346871429`, one `GPU_1xH100`, environment version 5; its raw Jobs 2.2 definition now points to the replacement snapshot and command.
- First infrastructure attempt: submit Job/run/task `101226423985784` / `606502744002397` / `1027372021943020`; terminal `FAILED`/`INTERNAL_ERROR`. H100 provisioning and dependency installation completed, but the generated command selected AppleDouble file `/databricks/code_source/._air` as its working directory. User code lasted one second, with zero generations, PPO updates, or experiment results. MLflow run `0fba5f6d8c5146d6aa24355f39d6ec05` is failed infrastructure evidence only.
- AIR-CLI replacement Job/run/task: `623913130478004` / `944543825231997` / `48349164926518`; terminal `FAILED`/`INTERNAL_ERROR` on attempt zero. One H100 was allocated, the environment installed all dependencies in 36 seconds, and user command execution lasted one second. Databricks CLI 1.9's experimental AIR path uploaded the correct source but dropped `ai_runtime_task.code_source_path` from its Jobs payload, so the launcher reported `CODE_SOURCE_PATH: unbound variable`. It produced zero model generations or PPO updates. MLflow run `f6750f63efe043d6a5ce771bf5fd906b` contains the terminal launcher/application logs and system metrics only.
- First direct persistent Job run/task: `7802116005791` / `412046908720153` under Job `683906346871429`; terminal `FAILED`/`INTERNAL_ERROR` on attempt zero. The source mounted correctly, the protocol and 12/12 group-disjoint corpus resolved, CUDA reported `NVIDIA H100 80GB HBM3`, and all frozen parameters were logged. AIR then failed on the first tokenizer download because it sets `HF_HUB_ENABLE_HF_TRANSFER=1` while `hf_transfer` was absent. User code ran 20 seconds; there were zero generations, baseline scores, or PPO updates. MLflow run `65e0e1bd46ee41d198b2459d11c00d00` contains the exact traceback and pre-generation tags/parameters.
- Dependency-amended persistent Job run/task: `1024235606646657` / `1033390266608877`; terminal `FAILED`/`INTERNAL_ERROR` on attempt zero, MLflow `e761b9770593429ebc7b7ecb1b72a23d`. The H100, source mount, model download, and complete deterministic baseline succeeded: `1/12` held-out full repairs, `15/36` hidden cases, and zero unsafe outputs. The first sampled PPO batch was generated and scored, but `trainer.step()` failed before changing weights because TRL 0.11.4's `RunningMoments.update()` stores `mean/std` as Python floats while `PPOTrainer.step()` calls `.to()` on them. Thus this run performed zero PPO updates; its baseline is diagnostic, not the paired result.
- Completed TRL-compatibility persistent Job run/task: `622161538716123` / `320447701125649`; terminal `SUCCESS` on attempt zero after 1,845 seconds, MLflow `5517069a75334139ba8a7b8e27417828` ended `FINISHED`. The source adds only a tensor-type compatibility shim around TRL's running-moment update, preserving score scaling and normalization. Jobs idempotency token: `4f94f41c9d280d946c918737be9fd8e40e2c3058d495e3045d4dd97139f81599`.
- Active source snapshot: `/Workspace/Users/austin.choi@databricks.com/.air/repo_snapshots/codex_hydrogym/codex_hydrogym_20260827_204215.tar.gz`, raw SHA-256 `1c411c2f15aced55b8f960e796c6c199a2b4d2d494c04e8c826233bd3839decf`. Its exact five members are `codex_hydrogym/__init__.py`, `tracking.py`, `coding_rl/__init__.py`, `coding_rl/experiment.py`, and `agent_eval/CODING_AGENT_PPO_PROTOCOL.md`; the exported remote bytes and member list were verified before launch.
- Replacement command: `/Workspace/Users/austin.choi@databricks.com/.air/cli_launch/codex_hydrogym/coding_agent_ppo_real_v1_40b57ed6a94f48f5/command.sh`, containing `cd "$CODE_SOURCE_PATH/.." && python -m codex_hydrogym.coding_rl.experiment`.
- Active hashes: protocol `7df7184c900197357106063a5739f2b06f48e1965f49823031d0f62f59b07c54`, experiment `31c5c963a45333e93e72153cad31b74cfe4b531a540683e5e7f473a90226086c`, workload `8d17f611999dd0669e418f2cbcce10caefd93754df132e8570c8e4d8030e75f6`; combined key `64d4968189a275038aa26da50a9227ce6e6131e73c2b8e8292bac4bda4a7f537`.
- Replacement idempotency key: `244421448b5454252a9fa7cd4e24e3704927f9c2883fcd0be37f4b6cf639889e`. `COPYFILE_DISABLE=1` was set for both the non-executing AIR render and real submission.
- Terminal result: base held-out exact repairs `1/12` versus PPO `0/12`; base hidden cases `15/36` (`41.6667%`) versus PPO `7/36` (`19.4444%`); unsafe outputs `0` versus `0`. Deltas were `-0.0833333` full-repair rate and `-0.2222222` hidden-case rate, so `exploratory_positive=false`.
- Training completed all 24 updates and 192 rollouts in `339.1041` seconds of experiment time. Final-step KL was `0.6265215`; only 3 of 24 sampled batches contained any full repair, average batch reward was `-0.8645833`, and the final policy repaired `0/12` training tasks with one unsafe output.
- The saved adapter manifest contains ten files, including `adapter_model.safetensors` (8,676,008 bytes; SHA-256 `5cb21bd4fea3b8735106a125fe9c15ac5f94ba96ad24e5e6532092073735caa0`) and `value_head.pt` (5,226 bytes; SHA-256 `dca5b72d11e6ab52317063dd907043b8ab8cf6098dc3d55118f49aedc0d78dbf`).
- MLflow contains the protocol, corpus manifest, base/post records, 24 training rows, isolated snapshots, summary, and adapter. AIR could not upload native span bodies to its presigned `us-east-1.storage.cloud.databricks.com` URLs (`connection refused`); the 36 evaluation records retain unique trace IDs, but `mlflow.get_trace()` reports missing span data. These records are valid result artifacts but not a trace-native MemAlign labeling source.
- MLflow MemAlign was not run: its implementation rejects anything other than attributable HUMAN feedback. Do not relabel deterministic or model judgments as HUMAN. This experiment proves the end-to-end PPO weight-update path but provides evidence against benefit for this configuration.

The real experiment source is `codex_hydrogym/coding_rl/experiment.py`; workload and persistent-Job records are `codex_hydrogym/deploy/air/workload.coding-agent-ppo-v1.yaml`, `codex_hydrogym/deploy/coding_agent_ppo_v1_job.json`, and `codex_hydrogym/deploy/coding_agent_ppo_v1_job_update.json`. The review notebook now embeds the terminal result and was re-uploaded as the same object `2854344201905376`; local and exported source SHA-256 are both `991dd7e508475859065da8445c82457793071192d5ad83fea1e72c092908d66e`. A final read-only check with profile `dais-demo` reconfirmed Job run `622161538716123` as `SUCCESS`, task attempt zero as `SUCCESS`, MLflow run `5517069a75334139ba8a7b8e27417828` as `FINISHED`, and the H100/CUDA tags plus negative frozen metrics. No local tests or local training were run.

## Separate coding-agent quality proof now authorized

On August 26, 2026, the user explicitly authorized a separate, non-CFD agent-quality experiment. This does not reopen Gate 0 v3 and cannot authorize PPO or support a fluid-improvement claim. Its frozen purpose is: given identical HydroGym RunBundle evidence and an exact initial `AgentFeedback` draft, test whether a coding-model revision improves bounded reward-review quality, then test whether advice from a genuinely MemAlign-aligned `critic_quality` reviewer improves the same-model revision further.

The first Databricks milestone, a five-group sanity experiment, completed on August 26, 2026. It is pipeline validation, not a comparative proof. Each group preserved one exact draft and these conditions:

1. unchanged initial draft;
2. the same coding model revising that draft with advice from the registered base `critic_quality` reviewer;
3. a reserved MemAlign condition that remains unexecuted until attributable HUMAN `critic_quality` labels train the reviewer on a locked group-disjoint fold.

The coding-model transport is `system.ai.gpt-5-6-sol` through the authenticated `dais-demo` Unity AI Gateway. This is explicitly a read-only coding-model proxy, not the official OpenAI Codex SDK: the workspace currently has no OpenAI credential secret, and `dais-demo` authentication alone cannot authorize Codex. The official Codex SDK condition remains a later replication once separate OpenAI authority is supplied without copying local credentials into the workspace.

Evaluation is MLflow-native: managed datasets via `mlflow.genai.datasets.create_dataset()`, registered LLM judges, `mlflow.genai.evaluate()`, and MLflow traces. Claude Opus 5 is the advice-producing reviewer and may not score its own revision outcome. DeepSeek V4 Pro is the separately registered independent audit judge; deterministic schema/bounds/evidence/claim checks are additional sanity endpoints. The decision-bearing endpoint is blinded, held-out HUMAN `critic_quality`, grouped by evidence bundle. Five groups can validate transport and surface directional results only; no coding-agent or MemAlign benefit may be claimed until a prospectively locked 50-100+ group study and held-out human labels support it.

The completed sanity comparison was neutral at the ceiling: independent audit `5.0 -> 5.0`, issue coverage `0.9 -> 0.9`, and all three contract/safety endpoints `1.0 -> 1.0`; every group-level paired delta was zero, `directional_sanity_improvement=false`, and `safety_regression=false`. This proves that coding-model revision, MLflow evaluation, trace-native managed-dataset publication, and HUMAN-review plumbing operate end to end. It does not prove that the coding agent or MemAlign improves quality.

No CFD, solver execution, reward compilation into an environment, PPO, GEPA, prompt promotion, or controller promotion is permitted in this experiment. The legacy MemAlign Job `1046052441090117` remains forbidden because it targets `fluid_reward_plausibility` rather than this grouped `critic_quality` protocol.

Do not convert the positive development replication—or the extremely small v3 miss—into a Gate 0 or RL pass. The locked gate was conjunctive and failed.

`STATUS_REPORT.md` is the detailed source of record. This file is the operational handoff for continuing safely.

## Non-negotiable scientific boundaries

1. HydroGym computes executable rewards deterministically. A coding agent may propose or revise a bounded RewardSpec; it must never execute reward code inside the environment.
2. MemAlign can improve agreement of a `critic_quality` reviewer with expert judgment. It cannot improve or certify fluid dynamics directly.
3. No fluid-controller PPO or learned fluid-policy training occurs until an independently held-out, preregistered full Gate 0 passes. Gate 0 v3 executed once and failed; its near miss is not fluid-PPO authorization. The separately authorized coding-policy PPO experiment cannot override this boundary.
4. Never relax thresholds, drop difficult arms, append cases to a completed study, or reinterpret a failed screen after observing its result.
5. Preserve every frozen artifact and its source implementation. New scientific work gets a new study ID, source file, protocol fingerprint, and evidence directory.
6. A passing development diagnostic only permits designing a separately frozen held-out gate. It is not itself a Gate 0 pass.
7. Failure is an acceptable terminal result. Do not force a success for the demo.

## Terminal v3 audit completed

The read-only source notebook `codex_hydrogym/notebooks/re100_v3_terminal_audit.py` and persistent serverless Job definition `codex_hydrogym/deploy/re100_v3_terminal_audit_job.json` were prepared on August 26, 2026. The notebook is a separate standard-library implementation: it imports neither the production analyzer nor CFD/ML libraries and is designed to verify all seven terminal namespace artifacts, 360 exact traces, 720 windows, 2,520 numerical-gate values, 60 condition-level primary-gate values, paired state/history identities, derangement marginals, all analysis arithmetic, and the single frozen failure. It is uploaded at `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/re100_v3_terminal_audit` as notebook object `4282786714316185`; persistent Job `372974263929719` points to it.

Initial audit run `425177540299676` reached all three condition artifacts, then reproduced the primary wrapper defect at `result.json`: the producer hashed nested seed-cluster maps while their keys were integers, but JSON reload converted the keys to strings. Numeric sort order before serialization differs from lexical string sort order after reload, so the immediate canonical round-trip check failed. This is the exact code path between writing `result.json` and writing `air_run_summary.json`, and therefore explains the primary's terminal wrapper failure. The first audit task `341744783331041` failed on this expected mismatch; Databricks unexpectedly started platform attempt `59678641313442` despite the Job's zero-retry task configuration, so the parent was explicitly canceled to avoid waste. Both attempts were read-only and ran zero CFD/RL.

Corrected audit run/task `933651969763400` / `1023830567796956` completed `SUCCESS` on attempt zero. Local and exported workspace notebook bytes both have SHA-256 `8bb4d76a7bb52bf961e53169c30f9386e530ea77e395a49e2d5d3c71547e9d19`. Raw condition SHA-256 values are base `6ca51a074fbfbb94b39997706d7bafebc946719196178bb0ab9dc65faa97f144`, temporal `3e531d598bfb960fb0a80e45f368a35ad9da5b1c25c181917cc1f7ea6ee7634f`, and spatial `014f579ed0347030e089f43ba6fc7939595fb3088aa8a81148fb2cc25e25d7c1`. The audit required ordinary canonical validation for every artifact except the known result defect, required exact result raw SHA-256 `04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934`, reconstructed the typed pre-serialization digest `97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636`, and independently measured the invalid post-JSON canonical digest `114d858967d1b6a22743c08e9c0d49fd4fae950c1c96bfa37202780bcba04636`. It reproduced all 360 traces, 720 windows, 2,520 numerical gates, 60 primary gates, every analysis value, and exactly one false predicate. CPU serverless was intentional because this JSON/hash/scalar-statistics work completed in 20 seconds and would not benefit from a GPU.

## Current evidence chronology

### Gate 0 v1: numerical failure

- Protocol fingerprint: `ab83f36f4ed9991320ac6e191bb6330d4c7fdc502430a4b65bf4c5fbbe598fd5`
- Executed failure evidence: `codex_hydrogym/evidence/gate0/ab83f36f4ed9-c6c1c6002c25/development_search_failure.json`
- Failure digest: `c0be6dd593c948fd57437996f9f81274ce11c7e1736338d3c00e3c7346c80140`
- Re=200 at `48×48` reached a retained spectral-tail fraction of `0.082028`, above the frozen `0.05` limit.
- The similarly named `ab83f36f4ed9-c757978ca1e5` directory contains only a pre-execution protocol and must not be cited as the executed failure.

### Gate 0 v2: primary causal pass, final convergence failure

- Protocol fingerprint: `2729e365a5712b824d4d1a2257ade19519aa454509a221790ddcabf2021b32a9`
- Implementation digest: `5cc598c6b6e6168ec78d80360326d8d595ada81f4c9790b4d1c86b67f28350c4`
- Evidence directory: `codex_hydrogym/evidence/gate0/2729e365a571-5cc598c6b6e6`
- Primary artifact digest: `9ac1987e99bdaf7936a4c45047bf48d3e0df522bf95e22df1bf90b7edcbbc676`
- Convergence attestation digest: `a7903f7fd7798118d6f27ace45cbaff043665d1aeee4f0a8fdd8ef6daa440d1d`
- Final failed-report digest: `e8a19fa835a991f233ed7087c31a7c947c6a61a9a3c04ceb935db2870f8dd463`

The primary comparison passed all 20 gates:

| Arm | Mean TKE | Mean RMS L2 effort |
|---|---:|---:|
| Privileged oracle | `0.729081` | `0.500000` |
| Signed feedback | `1.135629` | `0.334375` |
| Zero | `1.909851` | `0.000000` |
| Locked fixed | `1.912044` | `0.250000` |
| Observation deranged | `1.958062` | `0.334375` |

The final convergence stage failed:

| Refinement | Max arm change | Limit | Max effect drift | Limit |
|---|---:|---:|---:|---:|
| Temporal | `8.0715%` | `2%` | `0.053650` | `0.02` |
| Spatial | `5.3793%` | `5%` | `0.042492` | `0.03` |

This result remains failed even though the feedback arm itself was comparatively stable. The zero/reference arm was the largest source of point sensitivity.

### Fresh-seed ensemble diagnostic: strong signal, negative screen

Runner: `codex_hydrogym/gate0/ensemble_diagnostic.py`  
Tests: `test/codex_hydrogym/test_ensemble_diagnostic.py`  
Evidence: `codex_hydrogym/evidence/ensemble_diagnostic/19927dd9f42c-0df64047e2c1`

- Study fingerprint: `19927dd9f42cdcda5a7faf938a1a9da7814e4bb3031a1f15b08670ece6dd6caf`
- Implementation digest: `0df64047e2c1372b79fb92420c2c99f75213f3386c1fc06767cd31f203de674b`
- Protocol artifact digest: `da12d0149689a632acf2dd76b02d7dfe9954ac3db0d50f3070915730ca60abf5`
- Base condition digest: `2d3e9b41c329a5a0e9692a68c17319b01bf8cec97924f6123a751a045ea509f3`
- Temporal condition digest: `e819c1b2a65450253ab75476365dda28092fbd77e933639efa86ad6f34768860`
- Spatial condition digest: `86c508e165271419d7b63a86ecbb83e53fc4926acafc6a143b229a290d759439`
- Result digest: `e45eb7f6b19b52d6580462ef22e0efcc87a6cd435916f88694ef25e375777d9b`
- Frozen decision: `supports_designing_full_gate=false`

Frozen design:

- Re=100, float64
- development seeds `401, 503, 607, 709`
- development phases `0.0625, 0.5625`, an opposite-phase pair
- reserved, unopened seeds `907, 1009`
- reserved, unopened phases `0.1875, 0.6875`
- uncontrolled burn-in `100` intervals
- controller warmup `50` intervals
- two consecutive scoring windows of `100` intervals each
- zero and gain-2 signed-feedback arms only
- radial action bound `0.5`
- base `64×64, dt=0.002`
- temporal `64×64, dt=0.001`, old margins `2%` arm and `2 pp` effect
- spatial `96×96, dt=0.002`, old margins `5%` arm and `3 pp` effect
- inference clustered by the four independent seeds

Observed condition metrics:

| Condition | Zero TKE | Feedback TKE | Aggregate reduction | Seed-cluster effect 95% CI | Minimum block reduction |
|---|---:|---:|---:|---:|---:|
| Base | `1.869819` | `1.108867` | `40.6965%` | `[38.8662%, 42.2993%]` | `35.3819%` |
| Temporal | `1.871635` | `1.102617` | `41.0881%` | `[38.9286%, 43.0175%]` | `33.4377%` |
| Spatial | `1.939336` | `1.112311` | `42.6448%` | `[40.0240%, 45.1252%]` | `38.0000%` |

Signed feedback beat zero in all 48 seed × phase × window blocks. Every numerical gate passed. Both window means were material in every condition, and all seed-cluster 95% intervals were far above the `5%` effect floor.

Refinement metrics:

| Refinement | Max arm difference | Arm limit | Aggregate effect difference | Effect limit | Paired-seed effect-difference 90% CI | Point checks | Equivalence CI |
|---|---:|---:|---:|---:|---:|---|---|
| Temporal | `0.5637%` | `2%` | `0.3915 pp` | `2 pp` | `[-1.7387, +2.5192] pp` | Pass | **Fail** |
| Spatial | `3.7179%` | `5%` | `1.9483 pp` | `3 pp` | `[+0.3957, +3.5879] pp` | Pass | **Fail** |

Exactly two screening predicates were false:

- `temporal_effect_equivalence_ci_supported`
- `spatial_effect_equivalence_ci_supported`

The temporal upper interval missed by `0.5192 pp`; the spatial upper interval missed by `0.5879 pp`. This is a real negative result, not a process error.

## Evidence integrity already verified

The existing evidence was round-tripped without rerunning CFD, and a separate read-only audit verified:

- all five diagnostic artifact digests;
- all seven frozen implementation-file hashes;
- all 48 condition traces;
- all 48 paired seed/phase/window blocks;
- paired initial and control-start state identities;
- window continuity and recomputed means;
- exact reproduction of the stored analysis;
- exact links to the v2 protocol and failed final report.

The current runner can revalidate the completed directory because existing condition artifacts prevent simulation:

```bash
PYTHONPATH=. uv run python -m codex_hydrogym.gate0.ensemble_diagnostic \
  --stage run \
  --output-dir codex_hydrogym/evidence/ensemble_diagnostic/19927dd9f42c-0df64047e2c1
```

Do not edit `ensemble_diagnostic.py` or any artifact in that evidence directory. Its source is implementation-hash-bound. Implement future studies in a new sibling module.

## Completed replication-sizing diagnostic: positive development screen

The next defensible study is implemented and frozen separately from the completed four-seed diagnostic. The user explicitly authorized its execution on August 25, 2026. The local CPU process was stopped on user direction after the base condition completed, without opening its metrics. The blind artifact remains in place, is excluded from analysis, and is bound by `platform_transition.json`. The user then explicitly requested GPU capacity when it can reduce runtime; `execution_backend_amendment.json` binds the sole primary Databricks execution to one H100 with JAX float64 while preserving every seed, phase, threshold, and analysis rule.

Runner: `codex_hydrogym/gate0/ensemble_replication.py`  
Tests: `test/codex_hydrogym/test_ensemble_replication.py`  
Evidence: `codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4`

- Study fingerprint: `269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62`
- Implementation digest: `a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c`
- Protocol artifact digest: `3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87`
- Protocol status: `frozen_before_execution` (the immutable preregistration state)
- Local execution status: stopped; `condition_base.json` exists but was not opened or interpreted, has raw SHA-256 `59ccee255aea7b73c88f3e28bc64a8fca00aed7c681476b231a4a2831bcc9cfd`, and is ineligible for analysis
- Databricks execution status: all 120 frozen trajectories and all result artifacts completed in the sole eligible namespace
- Result artifact digest: `c783ea92679ad9c3d51fc44a612d15f9c2fa4b548c0dd4d1d99133ce3222e35a`
- Frozen decision: `supports_designing_full_gate=true`
- Outer Job status: run `767477134906347` was terminated by the low-GPU watchdog only after the scientific runner had completed and uploaded results; this wrapper failure does not change the independently audited artifact result

Frozen design:

1. Use the ten SHA-256-derived development seeds `1100085772, 619716833, 1680869979, 270788329, 1326527252, 625393611, 901546380, 1422036434, 373522063, 1374108181`.
2. Do not reuse the prior four seeds as confirmatory observations and do not append results to the completed study.
3. Keep seeds `907, 1009` and phases `0.1875, 0.6875` sealed.
4. Preserve the same Re, precision, development phase pair, burn-in, warmup, two windows, arms, controller, grids, time steps, numerical gates, materiality floor, and convergence/equivalence margins.
5. Analyze the 10 new seeds as one fixed sample. No interim looks, optional stopping, seed replacement, or extension after seeing results.
6. Persist an immutable protocol before any CFD execution and use resumable, content-addressed artifacts per condition.
7. The run contains `3 conditions × 10 seeds × 2 phases × 2 arms = 120` trajectories, about 2.5 times the completed diagnostic.

Why 10 is a reasonable planning target, not evidence:

- observed temporal paired-difference mean `0.003902746`, standard error `0.009046426` at `n=4`, implied seed SD approximately `0.018092852`;
- observed spatial paired-difference mean `0.019918142`, standard error `0.006782164` at `n=4`, implied seed SD approximately `0.013564328`;
- a plug-in calculation suggests roughly seven seeds could suffice if the observed means and variances remained unchanged;
- ten fresh seeds provide some buffer, especially for the spatial `3 pp` margin, but do not guarantee a pass.

The frozen protocol retained the following screening logic:

- all numerical gates pass;
- feedback beats zero in every seed/phase/window block;
- every consecutive-window mean effect is at least `5%`;
- every seed-cluster effect 95% CI has lower bound at least `5%`;
- temporal arm and aggregate-effect point differences are within `2%` and `2 pp`;
- spatial arm and aggregate-effect point differences are within `5%` and `3 pp`;
- the paired-seed 90% temporal effect-difference CI lies wholly inside `[-2, +2] pp`;
- the paired-seed 90% spatial effect-difference CI lies wholly inside `[-3, +3] pp`.

Observed Databricks results:

| Condition | Zero TKE | Feedback TKE | Aggregate reduction | Seed-cluster effect 95% CI | Minimum block reduction |
|---|---:|---:|---:|---:|---:|
| Base | `1.877142` | `1.102652` | `41.258989%` | `[40.675922%, 41.711009%]` | `35.055739%` |
| Temporal | `1.884812` | `1.103438` | `41.456355%` | `[40.205436%, 42.604081%]` | `35.730960%` |
| Spatial | `1.877282` | `1.100463` | `41.379982%` | `[40.042200%, 42.503612%]` | `32.779492%` |

| Refinement | Max arm difference | Aggregate effect difference | Paired-seed effect-difference 90% CI | Frozen result |
|---|---:|---:|---:|---|
| Temporal | `0.408635%` | `0.197365 pp` | `[-1.008531, +1.431117] pp` | Pass |
| Spatial | `0.198496%` | `0.120992 pp` | `[-1.040266, +1.199146] pp` | Pass |

Signed feedback beat zero in all 120 seed × phase × window condition blocks. All 840 stored numerical-gate values were true. No prior, local, or four-seed observation was pooled into this analysis.

Implementation state:

- the new study/schema ID and claim boundary are distinct from the completed diagnostic;
- the runner imports the hash-bound diagnostic execution and analysis contract without modifying it;
- the manifest binds the new source, prior runner, Gate 0 runner/protocol, and four physics sources;
- `run` fails unless the exact protocol was frozen first;
- completed condition artifacts are immutable and resumable, and the loader independently checks hashes, exact cases/arms, numerical-gate schema, window continuity and means, and paired initial/control-start states;
- seven focused runner tests, five notebook tests, and all 252 scoped Python tests pass;
- the protocol and all eight implementation hashes were independently revalidated after freezing.

The stopped local process used this command and output directory; do not resume it:

```bash
env JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m codex_hydrogym.gate0.ensemble_replication \
  --stage run \
  --output-dir codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4
```

The only eligible analysis namespace is now:

```text
/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/evidence/269507101a52-a5ab894e5ff4/databricks-primary-20260825
```

The AIR route is `codex_hydrogym/deploy/air/workload.gate0-h100.yaml`, with fail-closed validation in `codex_hydrogym/deploy/air/gate0_replication_entrypoint.py`. It requires one `GPU_1xH100`, JAX x64, the exact wheel and source hashes, and both transition artifacts before running any trajectory. This GPU authorization applies only to the frozen diagnostic, not PPO.

### Decision after the replication

The frozen development screen passed, so a separate full Gate 0 v3 was implemented in `codex_hydrogym/gate0/re100_v3.py`, frozen at `codex_hydrogym/evidence/gate0_v3/885ff77559da-17fd18a51e8b/protocol.json`, and explained in `codex_hydrogym/gate0/GATE0_V3_PROTOCOL_REVIEW.md`. It used 12 prospectively unopened seed clusters, reserved phases `0.1875, 0.6875`, all five causal arms, two scoring windows, and uncertainty-aware equivalence for all five effect pairs. It has now executed once and those cases are opened terminal evidence. Its study fingerprint is `885ff77559dadd18cc54d91a30ecb6a48477a4c2baed46fc728635ea3eae8b38`, implementation digest `17fd18a51e8bfb2e8b6d018e7fe824a9b68921fe38d62d231e6634d6203b9dfe`, and protocol artifact digest `024039795a851caa0a1ea77580983aa2c869d40d05564f0765fdc56f1920db3f`.

Zero-CFD Databricks notebook object `838420173929124`, Job/run/task `693336960365518` / `132154788305477` / `336953652096218`, passed all 23 protocol/source/analysis/fail-closed checks. The frozen protocol still correctly records its pre-execution state—`execution_authorized=false`, `reserved_cases_opened=false`, and `rl_training_performed=false`—while the later external attestation authorized the one completed execution without mutating that artifact.

The first v3-specific one-H100 AIR preflight attempt used Job/run/task `316596818779054` / `403428098499749` / `108711422687827` and MLflow run `06f57e430b3a435a883861eab9bc23ca`. It timed out after 60 minutes before the entry point started: there were no application logs, custom MLflow tags, or artifacts. A post-run workspace audit found only the frozen `protocol.json`, so zero CFD ran and no reserved case opened. Treat this as an H100 capacity/setup failure, not a Gate 0 result.

The second attempt changed only the control-plane timeout from 60 to 120 minutes. Job/run/task `1057054860106994` / `199802627560370` / `751301736884830` completed `SUCCESS`; MLflow run `3b16e54b500d49d6a36866d0343ce386` ended `FINISHED` and contains `gate0_v3/preflight.json` plus the frozen protocol. Its canonical payload confirms one `NVIDIA H100 80GB HBM3`, GPU-backed JAX/JAXLIB `0.7.2`, x64, all pinned packages and exact digests, `cfds_executed=0`, `execution_authorized=false`, `reserved_cases_opened=false`, and `rl_training_performed=false`. The post-run workspace audit again found only `protocol.json`. Entry-point SHA-256 is `0cd4f68cd6501c95b76f9fb204747f194eeebb8aa55606b87f08b9a67f5d68f6`; base workload SHA-256 is `2d263081c398fa1132e5a439a2b3cfc21cf23c65016c098279669389a616b706`.

On August 26, 2026 at `20:03:41Z`, the user explicitly approved the one full held-out execution. The resulting review attestation has artifact digest `39b4ab964755ff1ea1d7747939ac77517219faa2648702adbc4d022040902667` and raw SHA-256 `4dba8cab5f561ce3800915d3aaa0d5827f33afba526282d1f0b48f758c7a605d`. Its separate execution-token hash is `f6cf4be64101d4642489de6bd9c9558c67dfb152c3795c56471a3d0a237b51c5`; the token value exists only in Databricks secret `codex-hydrogym-gate0-v3/one-full-execution-885ff77559da`. The fail-closed primary AIR entry point SHA-256 is `7beb2b8c16428ffe9ca9e31cd418752ea3ad2a485259f17b6679e416eee44e9d`; authorization-preflight and primary-workload SHA-256 values are `27371bf1435692c350c096707ecdd756c0f10fe74428801f87b405250a3800bf` and `59d949d3474c8363cb43d162abbeaf5f8d0562ac0d2685eb1749419030c443ae`. Both AIR dry runs passed, and no reserved case was opened while creating or validating these artifacts.

The final read-only namespace audit immediately before authorization launch found only the frozen `protocol.json`. Authorization preflight Job/run/task `236084355460379` / `251966954943330` / `954992926132786` was submitted once at `2026-08-26T20:09:55Z` on `dais-demo` with digest-bound idempotency key `1315a90700407376b2d613c43d209a8b6977db973a80b6c0302b68aa8e955d2d`; it completed `SUCCESS` on attempt zero after 2,787 seconds. MLflow run `898748e8ef17461399f37ef746208541` ended `FINISHED` with the authorization payload, frozen protocol, and review attestation. The canonical payload confirms the exact token/attestation/digests, one H100, JAX x64, zero CFD, no prior/local observations, no opened reserved case, and no RL.

The final namespace audit immediately before primary launch again matched exactly the sole 14,109-byte frozen `protocol.json`. Primary Job/run/task `98243916406855` / `425683771687715` / `429856525625340` was submitted once at `2026-08-26T20:58:40Z` with digest-bound idempotency key `f0ffd5678ec43bb66e3bba03bcefae19043810eb24aa0bb97a3aac9e78fc2afb`. It used one H100, no scientific retries, 360 trajectories, 720 windows, and MLflow run `e7caeb85879c4aa988c5c39d05ed781d`.

After about 46 minutes of H100 provisioning, the primary MLflow claim-role/backend/study/workflow tags appeared. Those tags are set only after the fail-closed entry-point validation returns, so the exact token, attestation, source/wheel/protocol digests, package pins, one-H100/JAX-x64 runtime, and protocol-only namespace checks passed. No log, condition artifact, or partial metric was opened before terminal state.

Primary Job/run/task `98243916406855` / `425683771687715` / `429856525625340` reached terminal `FAILED`/`INTERNAL_ERROR` after 23,766 seconds; MLflow `e7caeb85879c4aa988c5c39d05ed781d` is also `FAILED`. All three condition artifacts and `result.json` were already written. The result contains all 360 trajectories and 720 windows, artifact digest `97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636`, and frozen decision `passed=false`. Exactly one predicate failed: temporal feedback-versus-zero 90% effect equivalence was `[-0.020255923821008402, 0.004795061435287584]` against the locked `[-0.02, +0.02]` region, a lower-bound miss of `0.0255923821` percentage points. Every base, causal, numerical, point-convergence, other temporal-equivalence, and spatial predicate passed. This remains a terminal Gate 0 v3 failure and does not authorize PPO. The wrapper failed after the result write but before `air_run_summary.json` and final MLflow upload; do not retry or extend the study.

Independent terminal audit notebook/job/run/task `4282786714316185` / `372974263929719` / `933651969763400` / `1023830567796956` completed `SUCCESS` on CPU serverless with zero CFD/RL. It reproduced every count and predicate and proved the wrapper root cause: 15 nested seed-cluster maps were hashed with integer keys before JSON converted them to strings, so the production runner's immediate post-write canonical validation failed. Exact result raw SHA-256 is `04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934`; typed pre-serialization digest is the stored `97f7002b…`; ordinary post-JSON canonical digest is `114d858967d1b6a22743c08e9c0d49fd4fae950c1c96bfa37202780bcba04636`. Do not repair or rewrite the frozen artifact; preserve the defect as evidence. Do not run v3 again.

## Work allowed only after a full Gate 0 pass

The existing PPO path does not implement the repaired Gate 0 task contract. Before any bounded training run, repair and test:

1. phase randomization inside vectorized episode resets instead of one fixed forcing phase;
2. the signed two-value forced-mode observation instead of the legacy `8×8` speed-grid observation;
3. policy packaging and serving signatures that currently assume `obs_size**2` inputs;
4. evaluation-context and checkpoint binding for the repaired observation/action contract;
5. zero, fixed-controller, and small reward-coefficient-grid baselines before PPO;
6. one bounded PPO trial with unchanged optimizer/compute settings across reward arms;
7. separate held-out evaluation with immutable evidence digests.

Only measured RunBundles from a valid task should feed the coding-agent reviewer experiment.

## Coding-agent and MemAlign state

- Frozen agent-revision protocol: `codex_hydrogym/agent_eval/AGENT_REVISION_PROTOCOL.md`.
- Reviewable source notebook: `codex_hydrogym/notebooks/coding_agent_memalign_proof.py`; workspace notebook object `4221023242892701` at `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_agent_proof/coding_agent_memalign_proof`.
- Persistent Databricks Job `1085457167903195`; its source model/evaluation run `884865696871903` completed all decision-bearing work before a downstream finalization error. Preflight `27ce7c6e171e46ff9285bf880c2df8b5`, Claude-advisor `5c7415b4b02649f4b5b0f782b78e8559`, scorer dry-run `3ff922e9f0064ae39a2a7e17df000f1b`, and DeepSeek paired-audit `af63f47baad445c29d744a2d19168886` are all `FINISHED` with no scorer errors.
- Zero-model-call finalizer Job/run/task `1085457167903195` / `412229067503407` / `89694089854172` completed `SUCCESS`. Summary MLflow run `c8c01383a8b546648e10d1c4c1f15c65` is `FINISHED`.
- Trace-native managed dataset `austin_choi_omni_agent_catalog.codex_hydrogym.agent_revision_sanity_v2` contains exactly ten records. The invalid v1 dataset/session remain excluded.
- HUMAN review session `codex_hydrogym_agent_revision_sanity_v2` is available at `https://fevm-austin-choi-omni-agent.cloud.databricks.com/ml/review-v2/9d1e4db611e449b1ac57d1faf3acba3c/tasks/labeling/8db20dc2-3ad5-42ac-a7f5-eb7020f01434?o=7474647489683936`.
- The five-group result is neutral: DeepSeek audit `5.0 -> 5.0`, preregistered issue coverage `0.9 -> 0.9`, contract/reward-safety/claim-scope `1.0 -> 1.0`, and every paired group delta zero. `directional_sanity_improvement=false`; `safety_regression=false`.
- This proves the coding-model revision and MLflow/HUMAN-review plumbing work. It does not prove coding-agent benefit. These synthetic groups may be labeled to validate the review UI but may not train a claim-bearing MemAlign reviewer.
- A separate project-specific repair pilot is frozen in `codex_hydrogym/agent_eval/CODING_AGENT_REAL_BUG_PROTOCOL.md` and implemented in `codex_hydrogym/notebooks/coding_agent_real_bug_proof.py`. It uses 12 historical defect families, four bounded edit choices per case, 36 hidden deterministic regression checks, and a predeclared direct-agent pass rule of at least 10 exact repairs, Wilson 95% lower bound above `0.50`, and zero unsafe selections. It does not execute model-generated code.
- The repair notebook is uploaded as object `2982479944637502` at `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_agent_proof/coding_agent_real_bug_proof`; persistent Job `538885349695793` points to it. Zero-model review run/task `532815540197813` / `576618998523022` completed `SUCCESS` and locked protocol fingerprint `7720c746c4ff556cf1f76684d830f96f5076d114e1ab2a252fbcf276f9da984f`.
- Decision-bearing parent run `690116555720697` had an unexpected Databricks platform retry despite task `max_retries=0`. Attempt-zero task `847202305895048` and MLflow model run `a6e91299ff084fb79f57cac67d77992e` stopped before deterministic scoring because one Claude rationale exceeded the 1,200-character harness guard. Automatic attempt-one task `35953586830762` completed `SUCCESS`; its model run `27833f359d824ddaaa15da2b64b55acb` and deterministic audit run `51afc04ff15a43bc85cbb2f8d4776aac` are `FINISHED`.
- The completed result passed its frozen direct-agent criterion: `12/12` exact minimal repairs, `36/36` regression checks, zero unsafe edits, exact-repair rate `1.0`, and Wilson 95% lower bound `0.7575059933447591 > 0.50`. The observed-bug baseline was `0/12`. Claude recommended keeping all 12 correct selections, so reviewed-versus-direct delta was exactly zero, `base_review_helped=false`, and there was no review safety regression.
- Managed dataset `austin_choi_omni_agent_catalog.codex_hydrogym.coding_agent_real_bug_pilot_v1` contains 24 native-trace deterministic-audit records. This corpus is not a `critic_quality` MemAlign training set.
- Independent zero-model audit notebook object `2982479944637503`, Job/run/task `532806483721171` / `217804453082956` / `744897616507906`, and MLflow run `b0d9a3710b0e4d778de4bdaf4f6015d7` completed `SUCCESS`. It proved both platform attempts independently selected the same correct `12/12` direct repairs; attempt zero had 12 direct and 12 review traces but no revision/audit condition, while attempt one had all 36 model traces and 24 scored records.
- This is positive evidence that the read-only GPT coding-model proxy was useful on the 12 frozen, bounded project-maintenance cases. It is not evidence that it can author/apply arbitrary patches, improve reward review, improve fluid control, or generalize to other repositories. Claude review and MemAlign benefit were not shown; GEPA remains deferred.
- Direct GPT and Claude transport passed one synthetic bundle only in remote MLflow run `b9908e32d1bb4cb4a633f10530f13bf5`.
- That run proves schema transport, not critic quality or fluid improvement.
- Its traces were created in a local MLflow store and serialized remotely; the target remote experiment contains zero retrievable native traces from that run.
- There are zero eligible adjudicated `critic_quality` labels.
- The current `align` CLI/job still targets legacy `fluid_reward_plausibility`, not the intended grouped `critic_quality` workflow.
- Alignment provenance, grouped train/held-out isolation, prompt-lineage rereads, and HUMAN/calibration artifact verification need repair before a measured reviewer experiment.
- Do not run MemAlign merely to create activity. It requires measured native traces, locked groups, and eligible human labels.

The intended later sequence remains:

`passing CPU Gate 0 → repaired PPO/evaluation contract → measured RunBundles → one Codex RewardSpec draft → base versus MemAlign reviewer advice → same-Codex paired revisions → human critic_quality labels and held-out reviewer evaluation → human approval → deterministic reward compilation → one bounded RL trial`

## App and Databricks state

- Databricks profile explicitly selected and used: `dais-demo`.
- Never auto-select a profile; pass `--profile dais-demo` only when the user has kept that choice in scope.
- The source-format notebook is uploaded at `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/ensemble_replication` (object `916404733159215`) with its reviewed wheel and both digest-bound transition artifacts.
- Notebook review Job `675379534762688`, run `1052166997412654`, succeeded remotely. Its structured output confirms the wheel/protocol/eight source hashes and both transition manifests, with `cfds_executed=0`.
- Persistent full AIR Job `236495542102189`, run `767477134906347`, completed the scientific runner and all 120 trajectories, then the outer task failed after 60 idle minutes with `Low Gpu utilization`. MLflow run: `77bee82ce31d4307845bab7c01bb8724`.
- Independent audit notebook `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/ensemble_replication_audit` is object `4286315606713468`. Audit Job/run/task-run `837419045027957` / `383118897658349` / `875720919604926` succeeded with zero CFD.
- The audit reproduced result digest `c783ea92679ad9c3d51fc44a612d15f9c2fa4b548c0dd4d1d99133ce3222e35a`, 120 traces, 240 windows, 840 numerical checks, and `supports_designing_full_gate=true`.
- The AIR wrapper now ends only MLflow runs it starts, with explicit `FINISHED`/`FAILED` status, and uses `result_artifact_digest` in a self-describing v2 summary while reserving `artifact_digest` for the summary itself.
- Zero-CFD engineering notebook object `838420173929111`, Job/run/task-run `1122866577108764` / `1112972985772444` / `633438677219642`, passed all 15 lifecycle and digest-schema checks against entry-point SHA-256 `eb9897c0ae977864ee6cd494b6a0d6157234247a1881492110bde7eb71252103`.
- Real one-H100 AI Runtime engineering preflight Job/run/task `345078684788904` / `784618286017300` / `130733335639096` completed `SUCCESS`. MLflow run `112972b1ea3f40f3be46d4e7b5442bea` ended `FINISHED` and contains `gate0/preflight.json`; the action was locked to `preflight`, so zero CFD trajectories ran.
- Gate 0 v3 protocol-review notebook object `838420173929124`, Job/run/task `693336960365518` / `132154788305477` / `336953652096218`, passed 23/23 zero-CFD checks.
- Gate 0 v3 H100 preflight attempt 1 Job/run/task `316596818779054` / `403428098499749` / `108711422687827` timed out before its entry point started. MLflow run `06f57e430b3a435a883861eab9bc23ca` ended `FAILED` with no preflight artifact; the v3 evidence directory still contains only `protocol.json`.
- Gate 0 v3 H100 preflight attempt 2 Job/run/task `1057054860106994` / `199802627560370` / `751301736884830` completed `SUCCESS`. MLflow run `3b16e54b500d49d6a36866d0343ce386` ended `FINISHED` with `gate0_v3/preflight.json`; one H100, JAX x64, and zero CFD were confirmed, and the v3 workspace evidence directory still contains only `protocol.json`.
- The user explicitly authorized the one held-out v3 execution. Review-attestation digest `39b4ab964755ff1ea1d7747939ac77517219faa2648702adbc4d022040902667`, external-token hash `f6cf4be64101d4642489de6bd9c9558c67dfb152c3795c56471a3d0a237b51c5`, and both full-execution AIR dry runs are frozen; the token value is held only in a Databricks secret.
- Gate 0 v3 zero-CFD authorization preflight Job/run/task `236084355460379` / `251966954943330` / `954992926132786` completed `SUCCESS`; MLflow `898748e8ef17461399f37ef746208541` is `FINISHED`, the exact authorization payload passed, and the post-run namespace still contains only `protocol.json`.
- The sole Gate 0 v3 primary completed all 360 trajectories but the frozen decision failed one temporal equivalence predicate by `0.0255923821` percentage points. The outer task then failed before final summary upload; do not retry, extend, relax, or start PPO.
- Gate 0 v3 terminal audit notebook object `4282786714316185`, persistent Job `372974263929719`, run/task `933651969763400` / `1023830567796956`, completed `SUCCESS` and reproduced 360 traces, 720 windows, 2,520 numerical gates, 60 primary gates, the exact failure, and the integer-key serialization root cause with zero CFD/RL.
- The notebook defaults to non-mutating `review`, has a zero-CFD `preflight`, requires an exact execution token, and treats the full Databricks namespace as the sole decision-bearing analysis set.
- The live app remains the obsolete deployment and must not be treated as reflecting the local containment/evidence work.
- The local AppKit copy still presents the v2 failure and four-seed negative screen; it has not yet been updated with or deployed for the positive ten-seed replication.
- The local contained app has no launch controls and remains undeployed.
- Do not broadly deploy the root bundle: it includes legacy jobs. A future deployment needs explicit user consent and a scoped app-only plan or separately isolated target.
- Do not start the bound feedback, MemAlign, GEPA, PPO, or reserved-case Gate 0 jobs.

## Validation state at handoff

The following Databricks validations passed on August 26, 2026 without local scientific execution:

```text
Independent completed-artifact audit: 120 traces, 240 windows, 840 numerical checks, zero CFD
AIR entry-point notebook validation: 15/15 checks, zero CFD
H100 engineering workload dry run: passed
Gate 0 v3 protocol review: 23/23 checks, zero CFD
Gate 0 v3 H100 preflight attempt 1: infrastructure timeout before entry point, zero CFD
Gate 0 v3 H100 preflight attempt 2: SUCCESS, one H100 and JAX x64 confirmed, zero CFD
Gate 0 v3 terminal independent audit: SUCCESS, 360 traces, 720 windows, 2,520 numerical gates, 60 primary gates, zero CFD/RL
Coding-agent sanity finalizer: SUCCESS, 10 trace-native records, HUMAN review session created, neutral paired result, zero CFD/PPO/MemAlign
Bounded real-bug repair pilot: PASS, 12/12 exact repairs and 36/36 checks with zero unsafe edits; independent zero-model audit SUCCESS; zero CFD/PPO/MemAlign
```

The following historical local checks passed on August 25, 2026; they were not rerun in this continuation because the user directed validation to Databricks:

```text
252 scoped Python tests
17 AppKit/Vitest tests
Ruff
TypeScript typecheck
ESLint
AppKit AST lint
Prettier
production server/client build
git diff --check
```

Commands:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q test/codex_hydrogym
.venv/bin/ruff check codex_hydrogym test/codex_hydrogym hydrogym/jax

cd codex_hydrogym/appkit/codex-hydrogym
npm test
npm run typecheck
npm run lint
npm run lint:ast-grep
npm run format -- --ignore-unknown
npm run build

cd ../../..
git diff --check
```

The production build emits only non-fatal warnings about stale Browserslist data and a large client chunk. Do not update dependencies merely to silence those warnings without a separate reason.

## Worktree caution

The repository is intentionally dirty and much of `codex_hydrogym/` is currently untracked from Git's perspective. Existing modifications belong to the user and prior work. At the last audit, `git status --short` included:

```text
 M .gitignore
 M hydrogym/jax/envs/kolmogorov.py
 M hydrogym/jax/solvers/base.py
 M hydrogym/jax/utils/utils.py
 M pyproject.toml
?? .isaac/
?? codex_hydrogym/
?? databricks.yml
?? hydrogym/jax/kolmogorov_contract.py
?? resources/
?? test/codex_hydrogym/
?? uv.lock
```

Do not reset, clean, overwrite, or commit unrelated work. Inspect before editing and use narrow patches.

## Immediate next-agent checklist

1. Read `codex_hydrogym/STATUS_REPORT.md` and this handoff completely.
2. Inspect `git status --short`; preserve all existing work.
3. Do not resume or inspect the quarantined local partial execution.
4. Review `codex_hydrogym/gate0/GATE0_V3_PROTOCOL_REVIEW.md`; the held-out seeds and phases were opened exactly once by primary run `425683771687715` and must never be reused as unseen cases.
5. Treat terminal audit run `933651969763400` as the independent source of record for v3 validation and the wrapper root cause. Do not rewrite `result.json`, rerun CFD, retry/extend v3, or relax its margin.
6. Stop before fluid-controller PPO: v3 executed once and failed. Any future fluid gate requires a new study ID, untouched cases, prospective rationale, separately frozen protocol, and explicit execution approval. The separate coding-policy PPO run does not authorize fluid training.
7. Distinguish the positive bounded maintenance result from the neutral reward-review sanity result. The former proves constrained repair selection on 12 frozen historical incidents only. A reward-review or MemAlign claim still needs a new prospectively locked, group-disjoint non-sanity study plus attributable train/held-out HUMAN `critic_quality` labels.
8. The Databricks App does not automatically ingest these artifacts and remains obsolete. Update or deploy it only under a separate explicit request.
9. Update `STATUS_REPORT.md` and this handoff after every completed work unit; preserve all dirty-worktree changes.

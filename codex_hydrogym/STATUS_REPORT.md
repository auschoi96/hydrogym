<!-- footer Databricks | codex_hydrogym | Implementation pause report -->
<!-- toc -->

# codex_hydrogym — Current Status and Evidence-Gate Report

**Status date:** August 27, 2026

**Workspace profile:** `dais-demo`

**Project label:** `codex_hydrogym`

**Current state:** Scientific Gate 0 still fails and no fluid-controller PPO has run. The separately authorized coding-policy PPO experiment completed `SUCCESS` on one Databricks H100, but its frozen directional result was negative: held-out exact repair fell from `1/12` to `0/12`, hidden-case coverage fell from `15/36` to `7/36`, and unsafe outputs remained zero. It proves the real PPO/LoRA weight-update path works; it does not prove RL or MemAlign benefit. Gate 0 v3 remains a terminal failure on one temporal equivalence predicate, and independent audit run `933651969763400` remains its source of truth. No threshold change, fluid PPO, GEPA, MemAlign training, model promotion, or contained-App deployment is authorized.

**Terminal coding-agent PPO result:** Protocol `codex_hydrogym.coding_agent_ppo.v1` compared the identical `Qwen/Qwen2.5-Coder-0.5B-Instruct` policy before and after 24 genuine TRL PPO/LoRA updates on 12 training and 12 group-disjoint held-out repair tasks. Persistent Job/run/task `683906346871429` / `622161538716123` / `320447701125649` completed `SUCCESS` on attempt zero in 1,845 seconds; MLflow run `5517069a75334139ba8a7b8e27417828` ended `FINISHED`. Experiment time was `339.1041` seconds for 192 rollouts. Base held-out exact repairs were `1/12`, hidden cases `15/36`, unsafe `0`; PPO values were `0/12`, `7/36`, unsafe `0`. The deltas were `-0.0833333` and `-0.2222222`, so `exploratory_positive=false`. Final-step KL was `0.6265215`; the final policy repaired `0/12` training tasks and emitted one unsafe training-evaluation output. A ten-file LoRA/value-head adapter was saved; `adapter_model.safetensors` is 8,676,008 bytes with SHA-256 `5cb21bd4fea3b8735106a125fe9c15ac5f94ba96ad24e5e6532092073735caa0`. Active source snapshot `/Workspace/Users/austin.choi@databricks.com/.air/repo_snapshots/codex_hydrogym/codex_hydrogym_20260827_204215.tar.gz` has SHA-256 `1c411c2f15aced55b8f960e796c6c199a2b4d2d494c04e8c826233bd3839decf`; protocol/experiment/workload hashes are `7df7184c900197357106063a5739f2b06f48e1965f49823031d0f62f59b07c54`, `31c5c963a45333e93e72153cad31b74cfe4b531a540683e5e7f473a90226086c`, and `8d17f611999dd0669e418f2cbcce10caefd93754df132e8570c8e4d8030e75f6`. The review notebook remains object `2854344201905376`; its local/exported SHA-256 is `991dd7e508475859065da8445c82457793071192d5ad83fea1e72c092908d66e` and it now displays the terminal result. A final read-only `dais-demo` check reconfirmed the Job and task as `SUCCESS`, MLflow as `FINISHED`, and the H100/CUDA tags plus frozen metrics. MemAlign did not run because no attributable HUMAN labels exist.

**Coding-PPO review links:** [notebook](https://fevm-austin-choi-omni-agent.cloud.databricks.com/editor/notebooks/2854344201905376?o=7474647489683936), [persistent Job](https://fevm-austin-choi-omni-agent.cloud.databricks.com/?o=7474647489683936#job/683906346871429), [successful run](https://fevm-austin-choi-omni-agent.cloud.databricks.com/?o=7474647489683936#job/683906346871429/run/622161538716123), and [MLflow result](https://fevm-austin-choi-omni-agent.cloud.databricks.com/ml/experiments/103455306564903/runs/5517069a75334139ba8a7b8e27417828?o=7474647489683936).

**New independent agent-quality lane:** The user explicitly authorized a non-CFD coding-agent experiment on August 26, 2026. It cannot change the Gate 0 decision or authorize PPO. The preregistered first milestone compared five grouped unchanged `AgentFeedback` drafts with same-model revisions informed by the registered base `critic_quality` reviewer. Claude Opus 5 produced advice and was excluded from outcome scoring; DeepSeek V4 Pro was the independent audit judge, alongside deterministic contract checks. A third MemAlign-advice condition remains reserved until real HUMAN `critic_quality` labels exist on a locked training fold. The primary future endpoint is blinded, group-held-out HUMAN `critic_quality`; five cases are pipeline validation only, and a comparative claim requires 50-100+ prospectively locked groups. The authenticated workspace has no OpenAI secret, so the immediate transport is the read-only `system.ai.gpt-5-6-sol` coding-model proxy through Unity AI Gateway, not the official Codex SDK. No synthetic label may be represented as HUMAN or MemAlign evidence.

**Agent-quality execution status:** Alias-corrected source run `884865696871903` completed all decision-bearing model work before a downstream finalization error. MLflow preflight `27ce7c6e171e46ff9285bf880c2df8b5`, Claude-advisor `5c7415b4b02649f4b5b0f782b78e8559`, dry-run `3ff922e9f0064ae39a2a7e17df000f1b`, and DeepSeek paired-audit `af63f47baad445c29d744a2d19168886` are `FINISHED` with no scorer errors. Zero-model-call finalizer Job/run/task `1085457167903195` / `412229067503407` / `89694089854172` completed `SUCCESS`; summary MLflow run `c8c01383a8b546648e10d1c4c1f15c65` is `FINISHED`. Dataset `austin_choi_omni_agent_catalog.codex_hydrogym.agent_revision_sanity_v2` contains exactly ten native-trace records, and the HUMAN session is `https://fevm-austin-choi-omni-agent.cloud.databricks.com/ml/review-v2/9d1e4db611e449b1ac57d1faf3acba3c/tasks/labeling/8db20dc2-3ad5-42ac-a7f5-eb7020f01434?o=7474647489683936`. The result was neutral at the ceiling: audit `5.0 -> 5.0`, issue coverage `0.9 -> 0.9`, all three safety/contract means `1.0 -> 1.0`, and every group-level paired delta zero. `directional_sanity_improvement=false`; `safety_regression=false`. This proves the reward-review pipeline, not reward-review or MemAlign benefit. No eligible HUMAN label or MemAlign alignment exists.

**Bounded coding-maintenance result:** Protocol `codex_hydrogym.coding_agent_real_bug_pilot.v1` froze 12 historical project-defect families, four bounded repair choices per case, 36 hidden deterministic checks, and fingerprint `7720c746c4ff556cf1f76684d830f96f5076d114e1ab2a252fbcf276f9da984f` before model calls. Notebook object `2982479944637502`, persistent Job `538885349695793`, and zero-model review run/task `532815540197813` / `576618998523022` established the lock. Decision-bearing parent run `690116555720697` was automatically retried by Databricks despite task `max_retries=0`: attempt `847202305895048` failed before scoring because one Claude rationale exceeded 1,200 characters; automatic attempt `35953586830762` succeeded. Completed MLflow model/audit runs are `27833f359d824ddaaa15da2b64b55acb` / `51afc04ff15a43bc85cbb2f8d4776aac`. Direct GPT selected `12/12` exact minimal repairs, passed `36/36` checks, selected zero unsafe edits, and achieved Wilson 95% lower bound `0.7575059933447591`, passing the preregistered `>=10/12`, `>0.50`, zero-unsafe rule. Claude kept all 12 correct choices, so review delta was zero and `base_review_helped=false`. Dataset `austin_choi_omni_agent_catalog.codex_hydrogym.coding_agent_real_bug_pilot_v1` contains 24 native-trace scored records. Independent zero-model audit notebook object `2982479944637503`, Job/run/task `532806483721171` / `217804453082956` / `744897616507906`, MLflow run `b0d9a3710b0e4d778de4bdaf4f6015d7`, completed `SUCCESS` and proved that both platform attempts independently selected the same correct `12/12` direct repairs. This demonstrates bounded project-specific repair-selection utility, not arbitrary patch authorship, reward-review improvement, fluid improvement, or MemAlign benefit.

## Executive summary

The `codex_hydrogym` software and Databricks App foundation exists: a JAX Kolmogorov-flow path, PPO training code, deterministic reward compilation, MLflow tracking, a registered coding-agent revision prompt, human-review surfaces, Unity Catalog model-registration guards, and an AppKit visualization. Those components are useful infrastructure, but they do not establish that the current control problem is scientifically suitable for reinforcement learning or that model feedback improves it.

The confounded original task has been repaired enough to execute preregistered gates: the control curl now has the missing `2πi` derivative factor, the forcing has episode-specific phase, the forced mode has sine/cosine quadrature actuation, signed modal observations are versioned, and the evaluator includes phase/seed pairing plus an observation-derangement ablation. Those changes do not make the task valid by themselves. Frozen Gate 0 v1 fails numerically; Re=100 v2 passed all 20 primary gates but failed final temporal/spatial convergence; a four-seed development screen missed both uncertainty-aware intervals; and the later ten-seed development replication passed its frozen screen. The prospectively frozen v3 cases were then opened exactly once and v3 failed one temporal equivalence predicate. Those cases may never again be represented as unseen. The held-out PPO comparison remains blocked.

The architecture remains ordered as:

`passing CPU Gate 0 → repair the signed, phase-randomized PPO/evaluation contract → measured MLflow RunBundles → one Codex RewardSpec draft → base-versus-MemAlign reviewer advice → same-Codex paired revisions → human critic_quality labels and held-out reviewer evaluation → human approval → deterministic reward compilation → one bounded RL trial`

This remains a hypothesis, not a result, and there is no currently authorized path past its first step. Codex is the first coding-agent harness because its local transport passed and a single harness makes the paired reviewer experiment cheaper and cleaner. MemAlign may align only the composite `critic_quality` reviewer: it measures agreement with expert judgment, never fluid performance. HydroGym computes the approved reward deterministically during training; the coding agent never executes reward code in the environment. A small coefficient grid is the required cheap control. Claude and GEPA are optional later cost/quality benchmarks only if Codex plus the hand-written prompt measurably helps on held-out human judgments.

## Gate 0 v3 terminal audit — completed

Completed on Databricks on August 26, 2026:

- source notebook: `codex_hydrogym/notebooks/re100_v3_terminal_audit.py`;
- persistent Job definition: `codex_hydrogym/deploy/re100_v3_terminal_audit_job.json`;
- workspace notebook: `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/re100_v3_terminal_audit`, object `4282786714316185`;
- persistent serverless Job: `372974263929719`, configured with zero task retries;
- successful run/task: `933651969763400` / `1023830567796956`, `SUCCESS` on attempt zero in 20 seconds of task execution;
- notebook SHA-256: local source and exported workspace source both `8bb4d76a7bb52bf961e53169c30f9386e530ea77e395a49e2d5d3c71547e9d19`;
- compute: CPU serverless, because the workload is read-only JSON/hash/statistics validation and a GPU would not reduce the critical path;
- scope: seven exact artifacts, 360 trajectories, 720 windows, 2,520 numerical-gate booleans, 60 recomputed condition-level primary predicates, controller/derangement/history integrity, all means/effects/t-intervals/refinement predicates, and the exact terminal decision;
- exclusions: no production-analyzer import, CFD, JAX, NumPy, MLflow mutation, PPO, reward experiment, MemAlign experiment, or App deployment.

Initial diagnostic audit run `425177540299676` independently reproduced the primary wrapper failure. Task `341744783331041` validated the condition artifacts and stopped when the ordinary post-JSON canonical digest of `result.json` differed from its stored digest. Databricks started platform attempt `59678641313442` despite the zero-retry task setting; the parent was canceled after the deterministic defect was identified. Both attempts were read-only and executed zero CFD/RL.

Corrected audit run `933651969763400` then completed `SUCCESS` on attempt zero. It required ordinary canonical round trips for the protocol, review attestation, execution context, and all three condition artifacts. Raw condition SHA-256 values are base `6ca51a074fbfbb94b39997706d7bafebc946719196178bb0ab9dc65faa97f144`, temporal `3e531d598bfb960fb0a80e45f368a35ad9da5b1c25c181917cc1f7ea6ee7634f`, and spatial `014f579ed0347030e089f43ba6fc7939595fb3088aa8a81148fb2cc25e25d7c1`. It required exact result raw SHA-256 `04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934`; proved that the ordinary post-JSON canonical digest is `114d858967d1b6a22743c08e9c0d49fd4fae950c1c96bfa37202780bcba04636`, not the stored digest; and reconstructed the typed integer-key preimage that does yield stored digest `97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636`. The producer created 15 nested seed-cluster maps with integer keys, and JSON converted them to strings; numeric key sorting before serialization therefore differed from lexical sorting after reload. The production runner validates immediately after writing `result.json`, so this defect fully explains the missing `air_run_summary.json` and final MLflow artifact upload. The independently recomputed science remains `passed=false` on exactly `temporal_all_effect_equivalence_intervals_inside_margin`.

## Live resources

Read-only workspace inspection authenticated successfully with profile `dais-demo` on August 24, 2026.

| Resource | Identifier | Current status |
|---|---|---|
| Databricks App | [codex_hydrogym control cockpit](https://codex-hydrogym-7474647489683936.aws.databricksapps.com) | `ACTIVE`, but its UI still advertises the obsolete bootstrap → MemAlign → GEPA → H100 sequence |
| MLflow experiment | `/Shared/codex_hydrogym` — `103455306564903` | Contains infrastructure diagnostics and completed coding-PPO run `5517069a75334139ba8a7b8e27417828`; no fluid-PPO result |
| Coding-policy PPO review notebook | Object `2854344201905376`; `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_agent_proof/coding_agent_ppo_real_experiment` | Review/launch surface now embeds terminal negative result; local/exported SHA-256 `991dd7e508475859065da8445c82457793071192d5ad83fea1e72c092908d66e` |
| Coding-policy PPO persistent Job | `683906346871429` | Raw Jobs 2.2 definition points to the final five-file snapshot, pinned environment, and one-H100 command |
| Coding-policy PPO attempt 1 | Job/run/task `101226423985784` / `606502744002397` / `1027372021943020` | `FAILED`/`INTERNAL_ERROR` before experiment code because AIR selected AppleDouble file `._air`; zero generations and zero PPO updates |
| Coding-policy PPO AIR-CLI replacement | Job/run/task `623913130478004` / `944543825231997` / `48349164926518` | `FAILED` after H100 allocation and 36-second dependency install because CLI serialization omitted `code_source_path`; zero generations/PPO updates, MLflow `f6750f63efe043d6a5ce771bf5fd906b` |
| Coding-policy PPO first direct persistent run | Job/run/task `683906346871429` / `7802116005791` / `412046908720153` | Source/H100/protocol checks passed; failed before generation because enabled fast transfer dependency was absent; MLflow `65e0e1bd46ee41d198b2459d11c00d00` |
| Coding-policy PPO dependency-amended run | Job/run/task `683906346871429` / `1024235606646657` / `1033390266608877` | Baseline `1/12`, `15/36`, zero unsafe; failed before first update on TRL float/tensor bug; MLflow `e761b9770593429ebc7b7ecb1b72a23d` |
| Coding-policy PPO TRL-compatibility run | Job/run/task `683906346871429` / `622161538716123` / `320447701125649` | `SUCCESS` on attempt zero; 24 updates/192 rollouts; base `1/12` vs PPO `0/12`, zero held-out unsafe; MLflow `5517069a75334139ba8a7b8e27417828` `FINISHED` |
| Coding-agent proof notebook | Object `4221023242892701`; `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_agent_proof/coding_agent_memalign_proof` | Reviewable MLflow-native source; current revision is being repaired for native judge endpoint names |
| Coding-agent proof Job | `1085457167903195` | Run `884865696871903` completed all model/evaluation traces; finalizer `83406823950834` created the exact ten-record trace-native v2 dataset, then stopped before session creation on an existing-schema overwrite; reuse/validation fix pending |
| Failed advisor evaluation | MLflow run `6598e8cd5e1a4023b99407fd104a06dd` | Five of five `critic_quality` assessments are null with `SCORER_ERROR` / `ENDPOINT_NOT_FOUND`; no revisions were made |
| UC revision prompt | `prompts:/austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_reward_revision/1` | Registered and verified; alias `baseline`; not yet consumed by a measured remote revision run |
| Feedback bootstrap job | `528338478292601` | Bound to the app; zero runs; earlier proposal-feedback path |
| MemAlign job | `1046052441090117` | Bound to the app; zero runs; legacy `fluid_reward_plausibility` definition, so do not run it as the current `critic_quality` workflow |
| GEPA job | `264942649048223` | Bound to the app; zero runs; legacy resource excluded from the MVP |
| Reusable H100 PPO job | `365523074984029` | Bound to the app; zero runs; do not run before Gate 0 passes |
| Replication notebook | `916404733159215` | Uploaded under the selected user's Workspace directory; remote review passed |
| Notebook review Job/run | `675379534762688` / `1052166997412654` | `SUCCESS`; verified wheel, protocol, eight source hashes, and transition records; zero CFD |
| Persistent replication AIR Job/run | `236495542102189` / `767477134906347` | Scientific runner completed all 120 trajectories and artifacts; outer task later failed the 60-minute low-GPU watchdog because its owned MLflow run was not ended |
| Replication MLflow run | `77bee82ce31d4307845bab7c01bb8724` | Contains completed replication artifacts; scientific result independently audited |
| Independent audit notebook | `4286315606713468` | Zero-CFD, standard-library-only audit notebook |
| Independent audit Job/run/task | `837419045027957` / `383118897658349` / `875720919604926` | `SUCCESS`; reproduced hashes, analysis, 120 traces, 240 windows, and 840 numerical checks |
| AIR engineering-validation notebook | `838420173929111` | Reviewable zero-CFD lifecycle and summary-schema validation |
| AIR engineering-validation Job/run/task | `1122866577108764` / `1112972985772444` / `633438677219642` | `SUCCESS`; all 15 checks passed against entry-point SHA-256 `eb9897…` |
| H100 engineering preflight | Job/run/task `345078684788904` / `784618286017300` / `130733335639096` | `SUCCESS`; MLflow run `112972b1ea3f40f3be46d4e7b5442bea` ended `FINISHED` with `gate0/preflight.json`; zero CFD |
| Gate 0 v3 protocol review | Notebook `838420173929124`; Job/run/task `693336960365518` / `132154788305477` / `336953652096218` | `SUCCESS`; all 23 zero-CFD protocol/source/analysis/authorization checks passed |
| Gate 0 v3 H100 preflight attempt 1 | Job/run/task `316596818779054` / `403428098499749` / `108711422687827` | `TIMEDOUT` after 60 minutes before the entry point started; MLflow run `06f57e430b3a435a883861eab9bc23ca` ended `FAILED` with no custom tags or artifacts; zero CFD |
| Gate 0 v3 H100 preflight attempt 2 | Job/run/task `1057054860106994` / `199802627560370` / `751301736884830` | `SUCCESS`; MLflow run `3b16e54b500d49d6a36866d0343ce386` ended `FINISHED` with `gate0_v3/preflight.json`; one H100, JAX x64, exact digests, and zero CFD confirmed |
| Gate 0 v3 authorization preflight | Job/run/task `236084355460379` / `251966954943330` / `954992926132786` | `SUCCESS`; MLflow run `898748e8ef17461399f37ef746208541` ended `FINISHED`; exact attestation/token/digests, one H100, JAX x64, zero CFD, and protocol-only namespace confirmed |
| Gate 0 v3 primary | Job/run/task `98243916406855` / `425683771687715` / `429856525625340` | All 360 trajectories and `result.json` completed; frozen decision `passed=false` on one temporal equivalence predicate. Outer AIR task then ended `FAILED`/`INTERNAL_ERROR` before summary upload; no retry is permitted |
| Gate 0 v3 terminal audit | Notebook `4282786714316185`; persistent Job `372974263929719`; run/task `933651969763400` / `1023830567796956` | `SUCCESS`; standard-library-only independent reproduction of 360 traces, 720 windows, 2,520 numerical gates, 60 primary gates, all statistics, the sole false predicate, and the integer-key serialization root cause; zero CFD/RL |
| Historical H100 job run | `366806901874578` from deleted job `278222053449531` | Failed during setup before user code; not a baseline result |
| Registered model target | `austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_ppo_controller` | No eligible version registered yet |

## Work completed

### HydroGym and PPO execution path

- Added a labeled `codex_hydrogym` training package around HydroGym's JAX Kolmogorov-flow environment.
- Implemented a bounded PPO configuration, rollout/training loop, checkpoints, validation artifacts, and deterministic numerical and physics gates.
- Built a bounded H100 workload path, now held behind the cheaper CPU scientific gate described below.
- Tagged baseline and aligned runs separately while keeping both in the same MLflow experiment.
- Required matching evaluation-context fingerprints before baseline-versus-aligned comparisons are considered valid.
- Added a separate frozen-training fingerprint that includes optimizer, rollout, seed, and budget fields while excluding only reward choice and run labeling.
- Added a deterministic reward wrapper around the existing HydroGym metrics. The executable v2 formula is `-TKE/E_ref - λu·||a||₁/2 - λΔ·||a-a_prev||²/4`.

This path is not ready to run the repaired task. The PPO defaults still use one fixed forcing phase and the legacy `8×8` speed-grid observation, while Gate 0 uses phase-sensitive signed forced-mode observations. The policy packaging and serving signature also hardcode `obs_size**2` inputs, so changing training to the signed two-value observation would currently break model registration. A Gate 0 pass would authorize repairing and testing these contracts, not launching the existing H100 job.

### MLflow reproducibility and Unity Catalog

- Consolidated all PPO, human feedback, judge alignment, GEPA, and promotion work into the one managed experiment `/Shared/codex_hydrogym`.
- Added a regression test that fails if alignment jobs or the App stop using that single experiment resource.
- Added MLflow parameters, step metrics, physics-gate metrics, tags, checkpoints, and evidence artifacts to the training path.
- Implemented a custom MLflow policy-model packaging and Unity Catalog registration path.
- Registration and the `candidate` alias occur only after deterministic physics validation passes.
- The production-promotion reader now rejects `train/*` curves and unfinished runs and requires separately named `heldout/*` metrics, matching SHA-256 evaluation-context and frozen-training fingerprints, and a lowercase SHA-256 `heldout_evidence_digest`. Prompt and model promotion enforce the same frozen-training parity, model-version metadata carries the frozen-training fingerprint, verified promotion tags are written before the serving-impacting alias, and the alias mutation is last. No current evaluator computes and logs those metrics plus digest-bound protocol provenance, so production promotion remains fail-closed and unusable by design. Digest syntax alone is not artifact verification; the separate held-out evaluator must compute and independently verify the digest from its immutable evidence artifact, bind its separate evaluation run to the exact trained checkpoint/model version, and cross-check the logged metrics.
- RewardSpec v2 contains only `control_l1_weight ∈ [0.05, 1]` and `action_delta_l2_weight ∈ [0, 0.25]`, plus formula/evidence identity. PPO optimizer and compute settings are forbidden.
- Reward compilation binds the model proposal to development-only `E_ref`, calibration evidence, human approval, and canonical digests. AI Runtime rejects an `aligned` run unless its materialized PPO config exactly matches that compiled manifest; a prompt tag alone cannot unlock training.
- Removed the accidental extra experiment from active use; it was moved to MLflow Trash.

### Human-feedback and coding-agent revision path

- Retained the earlier proposal/GEPA assets as legacy, non-authorizing infrastructure.
- Added a strict reward-only AgentFeedback v2 contract while preserving a read-only parser for archived RewardSpec v1 outputs.
- Implemented `initial Codex draft → reviewer advice → same Codex model/adapter revision`, with one exact draft reused across base and aligned reviewer arms.
- Bound each reviewer assessment to the RunBundle evidence digest and exact initial-draft digest. Cross-bundle, cross-draft, changed-model, and changed-adapter revisions fail closed.
- Registered the prompt the coding agent actually consumes and traced requested/resolved prompt URI, version, template digest, rendered digest, draft digest, reviewer digest, treatment, model, and adapter.
- Fixed AI Runtime prompt validation so the exact fully qualified UC URI for that registered revision prompt is accepted while a foreign or unlabeled prompt leaf still fails closed.
- Added the exact `critic_quality` labeling-session and one-consensus-HUMAN-label helpers, disjoint-fold checks, Codex-only MemAlign support, and a helper that registers the aligned scorer as a new version rather than replacing the base scorer.
- Kept promotion separate from reviewer alignment so an improved judge or revision cannot claim fluid-control benefit without held-out RL evidence and deterministic physics gates.

The MVP does not run GEPA. MemAlign must not overwrite a shared scorer or align on every labeled trace. It may train only on a locked critic-quality training fold, leave a grouped held-out fold untouched, and align the single adjudicated 1–5 `critic_quality` assessment. There are currently zero eligible labels. Ten usable native remote traces now exist, but they are synthetic sanity fixtures and therefore cannot train a claim-bearing MemAlign reviewer.

The local reward path is now causally connected: a bounded RewardSpec can compile into the deterministic wrapper only after evidence calibration and human approval, and AI Runtime verifies the matching manifest. This proves wiring, not usefulness. No measured coding-agent revision, approved manifest, PPO run, or held-out evaluation has exercised that path. The legacy GEPA function still optimizes the older direct reward-student prompt, not the registered coding-agent revision prompt, and therefore remains outside the current experiment.

An independent fail-closed audit found that this is still prototype wiring rather than a trustworthy executable loop: the current `align` CLI/job invokes the legacy `fluid_reward_plausibility` path rather than the new `critic_quality` function; train/held-out isolation is not revalidated by `group_id` and `critic_fold` inside the alignment function; reviewer advice and resolved-prompt lineage are not re-read from MLflow; and approval/calibration digests remain caller assertions rather than independently verified HUMAN and calibration artifacts. Those gaps must be resolved before the first measured reviewer experiment, but implementing the full provenance service before Gate 0 would be premature.

### Current direct-critic direction

- GPT and Claude receive the byte-identical canonical `RunBundle` prompt through a no-tool direct Databricks AI Gateway adapter and return the same strict `AgentFeedback` schema.
- The direct adapter forwards a strict structured-output schema, records the reported model and finish reason, and accepts only an exact configured model ID or an explicitly supplied provider alias; it fails closed on missing or unexpected reported-model IDs and on response, arm, adapter, or feedback-identity mismatches.
- MLflow tracing uses an `AGENT` root span with `harness_call` and contract-validation child spans. Adapter identity is first-class provenance.
- Managed-dataset records now require a native top-level `TRACE` source and distinct trace IDs. Publication verifies that every trace is readable, belongs to the target experiment, has state `OK`, has the expected sole `AGENT` root, and agrees exactly with the record's trace provenance, RunBundle, and outputs. Paired records use arm-scoped opaque input IDs; the four-adapter screen uses adapter-scoped IDs. A local SQLite MLflow round-trip proved four adapter records remain trace-sourced and idempotent.
- Held-out critic reporting has only claim-scoped metrics: mean absolute error, tie-aware Spearman correlation, and within-bundle GPT-versus-Claude preference agreement. These quantify critic alignment, not fluid control.
- The offline sanity corpus remains synthetic and cannot enter a non-sanity fold. Its five-group, ten-record remote revision screen completed with zero deltas. A HUMAN labeling session exists for UI validation, but no claim-bearing label or MemAlign alignment has run.
- The AppKit review route and library helpers use the exact `critic_quality` target, but no non-sanity trace is eligible for claim-bearing review. The remote registered-prompt revision loop has executed successfully; it produced neutral plumbing evidence, not quality benefit.
- Direct Gateway critics remain a cheaper transport baseline. The first measured reviewer-revision experiment uses Codex only; adding Claude is justified only if there is a controlled incremental question.

### Proven P0 transport result

Read-only reinspection with profile `dais-demo` confirmed finished MLflow run [`b9908e32d1bb4cb4a633f10530f13bf5`](https://fevm-austin-choi-omni-agent.cloud.databricks.com/ml/experiments/103455306564903/runs/b9908e32d1bb4cb4a633f10530f13bf5) in experiment `103455306564903`.

| Adapter | Model | Result | Latency | Decision | Tool activity |
|---|---|---:|---:|---|---:|
| `codex_direct` | `system.ai.gpt-5-6-sol` | Strict contract passed | `9.693 s` | `stop` | `0` |
| `claude_direct` | `system.ai.claude-opus-5` | Strict contract passed | `23.025 s` | `stop` | `0` |
| `codex_sdk` | `gpt-5.6-luna`, `openai-codex==0.147.0` | Strict contract passed | `11.580 s` | `stop` | `0` |
| `claude_agent_sdk` | `databricks-claude-sonnet-4-6[1m]`, SDK `0.2.142` | Failed closed on unexpected `HookEventMessage` | — | — | rejected |

The two direct calls used prompt SHA-256 `98e95a5d73808c6c64c4b5b726cbe6b58d14c496696d0ebe7b12b6bfc774d8d1`, a `4,096` token cap, the same strict transport schema, and no tools. The provider-reported aliases were `gpt-5.6-sol` for `system.ai.gpt-5-6-sol` and `us.anthropic.claude-opus-5` for `system.ai.claude-opus-5`; future calls must opt into those exact aliases explicitly. The run stores `p0_manifest.json` and four serialized source-trace artifacts. It is durable transport evidence, but those P0 trace IDs came from a local MLflow store and are not retrievable as native traces in the remote experiment. They therefore cannot be relabeled as remote `TRACE` dataset lineage. Later Databricks-native sanity and repair-pilot runs do have retrievable remote traces, but they do not retroactively repair P0 lineage. Future measured calls must trace directly to the target experiment, and publication now fails closed unless each source trace is readable, target-experiment-owned, successful, and consistent with its dataset record.

This P0 proves only that two direct model families can consume the same synthetic bundle and satisfy the contract. It does not measure critic quality, establish a controlled model comparison, validate MemAlign, justify GEPA, train RL, or show fluid improvement. The SDK models were also different revisions from the direct models, so the SDK follow-up is a system check rather than a transport-only model comparison. No further Claude SDK repair is planned for the MVP.

### Scientific Gate 0 currently fails

The original control confound is now represented by an executable, non-RL Gate 0 rather than hand-waved away:

- `control_term` applies the physical `2πi` spectral derivative and has manufactured sine/cosine regression coverage.
- The periodic grid excludes the duplicate endpoint and the dealiasing mask uses explicit integer Fourier modes.
- Episode forcing phase, forced-mode sine/cosine action channels, and signed forced-mode observations have versioned contracts.
- A development-only search locks the observation-free constant and signed-feedback gain before held-out cases are opened.
- Held-out arms compare zero, fixed constant, privileged phase oracle, signed feedback, and a whole-trajectory phase-and-seed derangement while preserving global observation/action marginals and effort.
- Final success fails closed until temporal and spatial refinement attestations are digest-bound to the exact primary artifact.

Frozen v1 protocol fingerprint `ab83f36f4ed9991320ac6e191bb6330d4c7fdc502430a4b65bf4c5fbbe598fd5` has multiple implementation attestations. The `c757978ca1e5…` directory is only a `frozen_before_execution` protocol artifact and must not be cited as the executed failure. The durable failed development search is `ab83f36f4ed9-c6c1c6002c25/development_search_failure.json`, with failure digest `c0be6dd593c948fd57437996f9f81274ce11c7e1736338d3c00e3c7346c80140`; it records `spectral_tail_controlled` failures for the attempted candidates/cases but, by the v1 schema, stores booleans rather than the numeric tail. The exact diagnostic rerun observed Re=200 `48×48` maximum retained tail `0.082028`, exceeding `0.05`. That numeric diagnosis is reproducible probe evidence, not a field in the durable v1 failure JSON.

The exact phase-0/seed-7 Re=200 probes were:

| Grid | Tail peak | Zero TKE | Gain-2 feedback TKE |
|---|---:|---:|---:|
| `48×48` | `0.082028` | `3.228788` | `2.450107` |
| `64×64` | `0.040299` | `3.070390` | `2.410477` |
| `96×96` | `0.006753` | `3.479144` | `2.508411` |

The Re=200 `64→96` comparison fails the locked convergence tolerances: zero-arm TKE changes `13.31%` against a `5%` limit, feedback TKE changes `4.06%`, and the effect changes `0.0641` against a `0.03` limit. The failure may combine spatial error with finite-window chaotic variability, but distinguishing those requires a costlier ensemble. Merely promoting the primary grid to `64×64` is not defensible.

Re=100 v2 was explicitly approved, separately frozen, and executed exactly once. Protocol `offset_phase_fp64_re100_gate0_v2` has fingerprint `2729e365a5712b824d4d1a2257ade19519aa454509a221790ddcabf2021b32a9`, implementation digest `5cc598c6b6e6168ec78d80360326d8d595ada81f4c9790b4d1c86b67f28350c4`, and immutable evidence directory `2729e365a571-5cc598c6b6e6`. It changed only the protocol ID, Reynolds number, primary grid, and spatial-refinement grid from v1; all phases, seeds, horizons, candidates, action/observation contracts, and causal/numerical/convergence limits were preserved. Two regression tests enforce that boundary.

Development passed and locked fixed action `(0.1767767, 0.1767767, 0, 0)` plus signed-feedback gain `2.0`, under lock digest `90c324567e20ab28cec778ad5d5dd53734619f2e04496046d9208e9c229e24df`. The full 12-cell held-out primary comparison then passed all 20 registered gates:

| Primary arm | Mean TKE | Mean RMS L2 effort |
|---|---:|---:|
| Privileged oracle | `0.729081` | `0.500000` |
| Signed feedback | `1.135629` | `0.334375` |
| Zero | `1.909851` | `0.000000` |
| Locked fixed | `1.912044` | `0.250000` |
| Observation deranged | `1.958062` | `0.334375` |

Signed feedback beat zero, fixed, and deranged controls in each independent held-out seed block. Its effort matched the deranged control exactly, global observation/action marginals matched, and every primary numerical gate passed. This primary evidence is bound by artifact digest `9ac1987e99bdaf7936a4c45047bf48d3e0df522bf95e22df1bf90b7edcbbc676`. It is a valid causal result inside the sampled primary evaluator, but it is not a final Gate 0 pass.

The digest-bound convergence stage was terminal:

| Refinement | Zero TKE, primary → refined | Feedback TKE, primary → refined | Max arm change | Limit | Max effect drift | Limit |
|---|---:|---:|---:|---:|---:|---:|
| Temporal, `dt 0.002 → 0.001` at `64×64` | `1.978711 → 1.818999` | `1.126668 → 1.133318` | `8.0715%` | `2%` | `0.053650` | `0.02` |
| Spatial, `64×64 → 96×96` at `dt=0.002` | `1.978711 → 1.872269` | `1.126668 → 1.145616` | `5.3793%` | `5%` | `0.042492` | `0.03` |

Both refinements passed numerical validity and preserved all ten materiality, opposite-phase, and effort decisions. Signed-feedback TKE itself changed only `0.5903%` temporally and `1.6818%` spatially. However, the frozen contract measures all arms and effects, not only the preferred controller. Both arm-convergence and effect-convergence gates failed, and exact ordering changed from `oracle < feedback < deranged < zero < fixed` to `oracle < feedback < zero < fixed < deranged`. The convergence attestation digest is `a7903f7fd7798118d6f27ace45cbaff043665d1aeee4f0a8fdd8ef6daa440d1d`; the final failed report digest is `e8a19fa835a991f233ed7087c31a7c947c6a61a9a3c04ceb935db2870f8dd463`.

A post-hoc case-level localization, which is diagnostic rather than decision evidence, found maximum per-phase TKE changes of `14.98%` temporal and `11.40%` spatial for the zero arm, versus `1.74%` and `4.62%` for signed feedback. The oracle also changed as much as `8.16%` in an individual spatial phase. Because all numerical gates passed, this pattern is more consistent with finite-window/discretization sensitivity across the benchmark than with a corrupt trace or solely an unstable feedback controller. It does not negate any failed preregistered gate.

The earlier phase-0/seed-7 Re=100 design probe had suggested spatial convergence, but it did not represent the full preregistered convergence case set and cannot override this executed result. Relaxing tolerances, dropping unstable arms, or retrying after seeing these results would be post hoc. Gate 0 v2 therefore fails honestly.

The gate deliberately uses a static phase within each episode. The primary pass establishes only sampled moderate-Re causal controllability; the failed convergence stage prevents a final scientific claim. Neither stage proves that PPO can infer a phase randomized inside vectorized resets, and neither performs RL.

### Fresh-seed ensemble diagnostic is promising but screen-negative

After the v2 failure was frozen, a separate exploratory protocol tested whether seed-clustered averaging and consecutive windows made the unchanged convergence margins plausible. This was not Gate 0 v3. Study `re100_fresh_seed_windowed_convergence_diagnostic_v1` used fresh development seeds `401`, `503`, `607`, and `709`; one opposite-phase pair; two consecutive 100-interval scoring windows; and only paired zero versus gain-2 signed-feedback arms. It evaluated the same base `64×64, dt=0.002`, temporal `64×64, dt=0.001`, and spatial `96×96, dt=0.002` conditions. Reserved phases and seeds were not opened. The claim boundary forbids treating the study as held-out, RL, coding-agent, MemAlign, GEPA, or fluid-improvement evidence.

| Condition | Zero mean TKE | Feedback mean TKE | Aggregate reduction | Seed-cluster effect 95% CI | Minimum block reduction |
|---|---:|---:|---:|---:|---:|
| Base | `1.869819` | `1.108867` | `40.6965%` | `[38.8662%, 42.2993%]` | `35.3819%` |
| Temporal | `1.871635` | `1.102617` | `41.0881%` | `[38.9286%, 43.0175%]` | `33.4377%` |
| Spatial | `1.939336` | `1.112311` | `42.6448%` | `[40.0240%, 45.1252%]` | `38.0000%` |

Signed feedback beat zero in all 48 seed × phase × window blocks, both consecutive-window mean effects exceeded the frozen `5%` materiality floor in every condition, every seed-cluster 95% CI was wholly above that floor, and all stored numerical gates passed. The point estimates also met all four old refinement checks:

| Refinement | Max arm difference | Arm limit | Aggregate effect difference | Effect limit | Paired-seed effect-difference 90% CI | Point checks | Equivalence CI |
|---|---:|---:|---:|---:|---:|---|---|
| Temporal | `0.5637%` | `2%` | `0.3915 pp` | `2 pp` | `[-1.7387, +2.5192] pp` | Pass | **Fail** |
| Spatial | `3.7179%` | `5%` | `1.9483 pp` | `3 pp` | `[+0.3957, +3.5879] pp` | Pass | **Fail** |

The result is therefore narrow but unambiguously negative under its frozen rule: only `temporal_effect_equivalence_ci_supported` and `spatial_effect_equivalence_ci_supported` were false, so `supports_designing_full_gate=false`. The temporal interval exceeded its upper equivalence margin by `0.5192` percentage points, and the spatial interval exceeded its upper margin by `0.5879` percentage points. This changes the diagnosis from obvious point nonconvergence to unresolved seed-level uncertainty; it does not establish equivalence, reverse the v2 failure, or prove that four seeds were sufficient.

The study fingerprint is `19927dd9f42cdcda5a7faf938a1a9da7814e4bb3031a1f15b08670ece6dd6caf`, its implementation digest is `0df64047e2c1372b79fb92420c2c99f75213f3386c1fc06767cd31f203de674b`, and its immutable evidence directory is `19927dd9f42c-0df64047e2c1`. The result digest is `e45eb7f6b19b52d6580462ef22e0efcc87a6cd435916f88694ef25e375777d9b`. A no-simulation round trip and an independent audit reproduced the analysis; validated all five artifact digests, all seven frozen implementation hashes, all 48 traces, and all 48 paired blocks; and confirmed the links to the v2 protocol and failed final report.

Appending cases to this completed study or promoting its passing point estimates after seeing the confidence-interval misses would be post hoc. The separately frozen replication below used a new fixed sample and a distinct analysis; it did not append to or pool this study.

### Ten-seed Databricks replication completed and passed its frozen screen

The separate study `re100_fresh_seed_windowed_convergence_replication_v1` is implemented in `codex_hydrogym/gate0/ensemble_replication.py` and frozen at `codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4`. The user explicitly authorized execution on August 25, 2026. The local CPU process was stopped on user direction after `condition_base.json` completed. Its metrics were not opened; only its byte length, timestamp, and raw SHA-256 `59ccee255aea7b73c88f3e28bc64a8fca00aed7c681476b231a4a2831bcc9cfd` were recorded. `platform_transition.json` makes that partial execution ineligible for analysis and names a separate Databricks namespace as the sole analysis set.

- Study fingerprint: `269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62`
- Implementation digest: `a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c`
- Protocol artifact digest: `3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87`
- New development seeds: `1100085772, 619716833, 1680869979, 270788329, 1326527252, 625393611, 901546380, 1422036434, 373522063, 1374108181`
- Sealed cases preserved: seeds `907, 1009`; phases `0.1875, 0.6875`
- Fixed workload: `3 conditions × 10 seeds × 2 phases × 2 arms = 120` trajectories

The seeds are the first ten eligible values from a frozen SHA-256 counter derivation after excluding all v1/v2, prior-diagnostic, and reserved seeds. The protocol retains Re=100 float64, phases `0.0625, 0.5625`, the 100-interval uncontrolled burn-in, 50-interval controller warmup, two 100-interval scoring windows, zero and gain-2 signed-feedback arms, the `0.5` radial bound, all three grids/time steps, all numerical and materiality gates, the `2%/2 pp` temporal margins, the `5%/3 pp` spatial margins, and the same point and uncertainty-aware screening predicates. The prior four observations are planning context only and are explicitly excluded from the replication analysis.

Execution was fail-closed unless the exact protocol already existed. It was resumable only by whole immutable condition artifacts, and its loader independently validated artifact/source identities, exact cases and arms, numerical-gate schema, interval/window reproduction and continuity, and paired initial/control-start states. The sole eligible Databricks namespace completed all 120 trajectories. Result artifact digest: `c783ea92679ad9c3d51fc44a612d15f9c2fa4b548c0dd4d1d99133ce3222e35a`.

| Condition | Zero TKE | Feedback TKE | Aggregate reduction | Seed-cluster effect 95% CI | Minimum block reduction |
|---|---:|---:|---:|---:|---:|
| Base | `1.877142` | `1.102652` | `41.258989%` | `[40.675922%, 41.711009%]` | `35.055739%` |
| Temporal | `1.884812` | `1.103438` | `41.456355%` | `[40.205436%, 42.604081%]` | `35.730960%` |
| Spatial | `1.877282` | `1.100463` | `41.379982%` | `[40.042200%, 42.503612%]` | `32.779492%` |

The temporal maximum arm difference was `0.408635%`, aggregate effect difference was `0.197365 pp`, and paired-seed 90% interval was `[-1.008531,+1.431117] pp`. Spatial values were `0.198496%`, `0.120992 pp`, and `[-1.040266,+1.199146] pp`. All ten frozen screening predicates passed, signed feedback beat zero in every one of the 120 seed × phase × window condition blocks, and all 840 stored numerical-gate values were true. The frozen result is `supports_designing_full_gate=true`.

Independent audit Job `837419045027957`, run `383118897658349`, task run `875720919604926`, executed notebook object `4286315606713468` and completed `SUCCESS`. The standard-library-only audit imported neither the solver nor production analyzer and ran zero CFD. It independently reproduced all artifact/source hashes, trace and window arithmetic, confidence intervals, convergence predicates, and final decision, while confirming zero prior/local observations in the analysis.

This result supports designing a full held-out Gate 0 v3. It does not establish held-out generalization, pass Gate 0, train a policy, or show RL improvement.

### Databricks notebook is uploaded and remotely reviewed

`codex_hydrogym/notebooks/ensemble_replication.py` is uploaded as notebook object `916404733159215`. Its 2,815,154-byte companion wheel has SHA-256 `91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1` and contains the exact protocol plus all eight implementation files. The platform-transition and execution-backend amendment artifacts are uploaded beside it.

The notebook defaults to `action=review`, which performs no installation or CFD. Review Job `675379534762688`, run `1052166997412654`, completed `SUCCESS` in Databricks and returned structured confirmation of the wheel SHA, protocol digest, all eight source hashes, the blind local exclusion, and the H100 amendment, with `cfds_executed=0`. `action=preflight` additionally checks Python 3.12, GPU-backed JAX, exactly one H100, float64, and every package pin. `action=run` requires the fingerprint-bound confirmation token and writes a decision-bearing execution context into the separate Databricks namespace.

The user explicitly requested GPU capacity when it can reduce wall-clock time. AIR does not offer CPU accelerators, so `execution_backend_amendment.json` authorized one `GPU_1xH100` while preserving float64 and the complete frozen protocol. Persistent Job `236495542102189`, run `767477134906347`, completed the scientific runner but the outer AI Runtime task later failed after 60 idle minutes because an MLflow run opened by the entry point was never explicitly ended. The low-GPU watchdog failure occurred after result upload and does not override the independently audited artifacts.

The wrapper now tracks MLflow ownership, ends only runs it starts, uses `FINISHED` or `FAILED` status, rejects bodies that shadow `artifact_digest`, and records the input as `result_artifact_digest` in explicit summary schema `codex_hydrogym.ensemble_replication_air_summary.v2`. Zero-CFD Databricks notebook run `1112972985772444` passed all 15 lifecycle/schema checks against source SHA-256 `eb9897c0ae977864ee6cd494b6a0d6157234247a1881492110bde7eb71252103`. Real H100 AI Runtime preflight run `784618286017300` then completed `SUCCESS`; its MLflow run `112972b1ea3f40f3be46d4e7b5442bea` ended `FINISHED` and contains `gate0/preflight.json`. This confirms the lifecycle repair in the runtime that previously idled into the watchdog, with zero CFD. The root bundle remains untouched because it contains legacy jobs outside this study's authorization boundary.

### Gate 0 v3 held-out execution and terminal audit

`codex_hydrogym/gate0/re100_v3.py` implemented the held-out design documented in `codex_hydrogym/gate0/GATE0_V3_PROTOCOL_REVIEW.md`: 12 prospectively frozen seed clusters, reserved phases `0.1875, 0.6875`, all five causal arms, two 100-interval scoring windows, all original primary gates, and paired-seed equivalence intervals for all five effect pairs. The 360 trajectories and 720 windows were executed exactly once on one H100 with JAX float64; these cases are now opened terminal evidence and cannot be reused as unseen cases.

- Study fingerprint: `885ff77559dadd18cc54d91a30ecb6a48477a4c2baed46fc728635ea3eae8b38`
- Implementation digest: `17fd18a51e8bfb2e8b6d018e7fe824a9b68921fe38d62d231e6634d6203b9dfe`
- Protocol artifact digest: `024039795a851caa0a1ea77580983aa2c869d40d05564f0765fdc56f1920db3f`
- Local artifact: `codex_hydrogym/evidence/gate0_v3/885ff77559da-17fd18a51e8b/protocol.json`
- Databricks artifact: `/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/evidence/885ff77559da-17fd18a51e8b/protocol.json`
- Reviewed wheel SHA-256: `e381b42d415b0644fd773be67ef9aab94133289e6559933bf72e079da50e2e51`

Zero-CFD Databricks notebook object `838420173929124`, Job/run/task `693336960365518` / `132154788305477` / `336953652096218`, passed all 23 independent seed, controller, source-hash, protocol, synthetic-analysis, and fail-closed authorization checks. The run stage refuses to proceed without a digest-bound human review attestation and matching separate token. The protocol records `execution_authorized=false`, `reserved_cases_opened=false`, and `rl_training_performed=false`.

The first v3-specific one-H100 preflight attempt used Job/run/task `316596818779054` / `403428098499749` / `108711422687827`. The rendered Jobs payload was an `ai_runtime_task` on exactly one `GPU_1xH100`, but Databricks timed the run out after 60 minutes before the entry point emitted any log or custom MLflow tag. MLflow run `06f57e430b3a435a883861eab9bc23ca` ended `FAILED` with no artifacts, and a post-run workspace listing found only the frozen `protocol.json`; no condition or result artifact exists. This is an infrastructure-capacity failure with zero CFD, not a Gate 0 result.

The retry changed only the control-plane timeout to 120 minutes. Job/run/task `1057054860106994` / `199802627560370` / `751301736884830` completed `SUCCESS` on attempt zero after H100 capacity became available. The runtime then spent 34 seconds preparing dependencies and 13 seconds in user code. MLflow run `3b16e54b500d49d6a36866d0343ce386` ended `FINISHED`; `gate0_v3/preflight.json` is 1,220 bytes and the logged frozen protocol is 14,109 bytes. The canonical payload confirms Python 3.12.3, one `NVIDIA H100 80GB HBM3`, GPU-backed JAX/JAXLIB `0.7.2`, x64, every package pin, the exact wheel/protocol/implementation digests, zero prior/local observations, `cfds_executed=0`, `execution_authorized=false`, `reserved_cases_opened=false`, and `rl_training_performed=false`. A post-run listing again found only `protocol.json` in the workspace evidence namespace. Entry-point SHA-256 is `0cd4f68cd6501c95b76f9fb204747f194eeebb8aa55606b87f08b9a67f5d68f6`; base workload SHA-256 is `2d263081c398fa1132e5a439a2b3cfc21cf23c65016c098279669389a616b706`.

The preflight completed before execution. The required digest-bound human review, separate execution token, final empty-namespace audit, and explicit user approval were then completed before any held-out trajectory ran.

Those remaining human-authorization artifacts were created after the user explicitly replied “yes do it” on August 26, 2026. Review attestation `codex_hydrogym/evidence/gate0_v3/885ff77559da-17fd18a51e8b/review_attestation.json` has artifact digest `39b4ab964755ff1ea1d7747939ac77517219faa2648702adbc4d022040902667` and raw SHA-256 `4dba8cab5f561ce3800915d3aaa0d5827f33afba526282d1f0b48f758c7a605d`. It binds the exact study, implementation, protocol, one H100, float64, 360 trajectories, 720 windows, successful preflight IDs, and one-execution decision. The separate token hash is `f6cf4be64101d4642489de6bd9c9558c67dfb152c3795c56471a3d0a237b51c5`; its value is stored only as Databricks secret `codex-hydrogym-gate0-v3/one-full-execution-885ff77559da`.

The new fail-closed primary entry point has SHA-256 `7beb2b8c16428ffe9ca9e31cd418752ea3ad2a485259f17b6679e416eee44e9d`. Authorization-preflight workload SHA-256 is `27371bf1435692c350c096707ecdd756c0f10fe74428801f87b405250a3800bf`; primary-workload SHA-256 is `59d949d3474c8363cb43d162abbeaf5f8d0562ac0d2685eb1749419030c443ae`. Both AIR dry runs passed and rendered exactly one H100, environment version 5, the Databricks secret reference, no retries, and the intended zero-versus-360 trajectory roles. No reserved case was opened during this authorization work.

Immediately before launch, a read-only `dais-demo` workspace audit again found exactly one namespace entry, the 14,109-byte frozen `protocol.json`, and AIR reported no active or historical runs for the current CLI identity. The zero-CFD authorization workload was submitted once at `2026-08-26T20:09:55Z` as Job/run/task `236084355460379` / `251966954943330` / `954992926132786`, with digest-bound idempotency key `1315a90700407376b2d613c43d209a8b6977db973a80b6c0302b68aa8e955d2d`. It completed `SUCCESS` on attempt zero after 2,787 seconds, of which the AIR application log attributes 33 seconds to environment setup and 15 seconds to user code. MLflow run `898748e8ef17461399f37ef746208541` ended `FINISHED` and logged `gate0_v3/authorization_preflight.json`, the frozen protocol, and the review attestation. The canonical payload confirms the external token and attestation, exact implementation/protocol/wheel hashes, one `NVIDIA H100 80GB HBM3`, JAX x64, `cfds_executed_before_runner=0`, no prior/local observations, `reserved_cases_opened_before_runner=false`, and `rl_training_performed=false`.

The final namespace audit immediately before primary submission again matched exactly the sole 14,109-byte frozen `protocol.json`. The one decision-bearing workload was submitted at `2026-08-26T20:58:40Z` as Job/run/task `98243916406855` / `425683771687715` / `429856525625340`, using digest-bound idempotency key `f0ffd5678ec43bb66e3bba03bcefae19043810eb24aa0bb97a3aac9e78fc2afb`. It ran on exactly one `GPU_1xH100`, with no scientific retries, a 24-hour timeout, 360 trajectories, and 720 windows. MLflow run `e7caeb85879c4aa988c5c39d05ed781d` is associated with that terminal execution.

After approximately 46 minutes of capacity provisioning, MLflow exposed the primary claim-role, H100-backend, study-fingerprint, and workflow tags. The entry point sets those tags only after its token, attestation, source/wheel/protocol digest, package, one-H100, JAX-x64, and protocol-only namespace validation returns successfully. This proves the fail-closed validation completed and the scientific runner entered its decision-bearing phase. No application log, condition artifact, or partial metric was opened.

The run reached terminal `FAILED`/`INTERNAL_ERROR` after 23,766 seconds. MLflow run `e7caeb85879c4aa988c5c39d05ed781d` also ended `FAILED`. The workspace namespace nevertheless contains immutable base, temporal, and spatial condition artifacts plus `result.json`, with exactly 360 trajectories and 720 windows. Result artifact digest is `97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636`; exported raw SHA-256 is `04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934`. Condition artifact digests are base `2d5f90d618de898b8457a0eeab695352a73521d2b6ea87153c8a823b1ae5a3e8`, temporal `8697ecbd559d147b88f61e2a8cb089561d0a65672979c96dc8314075b0156fb5`, and spatial `ac031fd7efd626737e4b2f9fcbcee0e90299b361b0d53b0cf9414ca2f806c3a4`.

The frozen scientific decision is `passed=false`. Exactly one screening predicate is false: `temporal_all_effect_equivalence_intervals_inside_margin`. The temporal feedback-versus-zero paired-seed 90% refined-minus-base effect interval is `[-0.020255923821008402, 0.004795061435287584]`, missing the locked `[-0.02, +0.02]` equivalence region by `0.000255923821008402` (`0.0255923821` percentage points) at the lower boundary. Every base causal and numerical predicate, every point-convergence predicate, all other temporal effect intervals, and all spatial predicates passed. This near miss is still a terminal Gate 0 v3 failure; no threshold relaxation, sample extension, retry, PPO execution, coding-agent reward experiment, or MemAlign claim is authorized.

Independent terminal audit Job/run/task `372974263929719` / `933651969763400` / `1023830567796956` completed `SUCCESS` with zero CFD/RL. It reproduced 360 traces, 720 windows, 2,520 numerical gates, all 60 condition-level primary gates, every condition mean/effect/t-interval/refinement predicate, and the single false predicate. It also resolved the wrapper error: the result producer hashed 15 nested seed-cluster maps while their keys were integers, but JSON reload converts keys to strings and changes sorted-key order. The raw result SHA-256 is `04c4d04782a507e7a04b878b8576b6186b7d1fff13c667db74d42a5e71091934`; the typed pre-serialization preimage reproduces stored digest `97f7002bf33e87beb02eef1a8f27f0ce73139641a6867f02cb61205ce67b9636`; the ordinary post-JSON canonical digest is instead `114d858967d1b6a22743c08e9c0d49fd4fae950c1c96bfa37202780bcba04636`. The production runner's immediate result round-trip validation therefore failed before `air_run_summary.json`, final metrics, and full artifact upload. This engineering defect neither invalidates the independently recomputed condition evidence nor changes its negative decision.

### Measured-local diagnostic, not a result claim

A sibling repaired tree at commit `f5eadecb2de4dc2812c7988adca5ce8bfd6b51e6` produced one useful CPU diagnostic with `dt=0.002`, seed 0, and a 512-step horizon:

| Controller | Settled TKE |
|---|---:|
| Zero action | `3.1696414947509766` |
| Constant opposition | `1.2599737644195557` |

The recorded fractional reduction was `0.6024869795192577`. The results record had SHA-256 `2c43041be0c1ad93f1fa1b0cb7523e86dfbb09b92af995be65397dd8aa0c69ce`.

This is diagnostic support for the fixed-forcing confound, not reproducible project evidence: it came from a different source tree, covered one seed, has no surviving raw log or MLflow run, and did not record the complete gate. It cannot support a fluid-improvement, RL, agent, MemAlign, or GEPA claim.

### Authenticated sibling GEPA/PPO evidence, not transferable to this lane

The supplied [GEPA review write-up](https://docs.google.com/document/d/16FWnnPpWN8oPsfGdRc9vQbXHmNXulgXMjvy8U1Euapk/edit) concerns the separate `claude_hydrogym` experiment `103455306563514`, not this project's experiment or current source tree. Read-only inspection with `dais-demo` nevertheless confirmed its central failure evidence:

- Finished PPO run `3954116690174b9492580de35267bc78` logged settled TKE `3.3466436863` for zero action, `1.2490334511` for constant opposition, and `1.4402053356` for the shipped learned policy. Its `tke_reduction_vs_oppose=-0.1530558564`, so the learned policy was 15.3% worse than constant opposition even though it beat zero action. The evaluation window was marked settled, but this was one training seed and lacks the current effort/phase protocol.
- GEPA job run `1027411304349635` and MLflow run `774fe7bcc5404cf6810c19f14bb820c9` were canceled after four proxy rounds, so they produced no final GEPA controller comparison.
- The four eight-seed proxy-round means for `tke_reduction_vs_zero` were `0.08979`, `0.08881`, `0.09508`, and `0.09241`. Their standard errors were `0.00658`, `0.01036`, `0.01109`, and `0.01226`; the full spread between round means was only `0.00627`. All 32 child trials logged `beats_oppose=0`.

This sibling result does not establish that GEPA fails generally, because its proxy, seeds, task formulation, and code are not the current protocol. It does show that the available GEPA run supplied no detectable benefit and ranked differences no larger than its sampling error. That is concrete cost/benefit evidence for excluding GEPA from the MVP and first fixing the experiment.

### Exact CPU Gate 0 contract

The locked comparison uses disjoint development and held-out phases/seeds, byte-identical developed initial states across arms, a radial action bound, separate TKE and effort accounting, opposite-phase materiality, independent seed-block wins, exact observation/action marginal checks, numerical gates, and temporal/spatial convergence. A failure in controllability, observability, causal ablation, effort, numerics, or convergence is a valid terminal result.

The runner now persists durable failure evidence: failed development searches record candidate, case, numerical-gate, partial-progress, and exception diagnostics without producing a controller lock; the explicit `convergence` stage requires and strictly loads an existing immutable primary report; repeated stages do not rerun completed evidence; and every refinement artifact binds to the protocol, lock, primary digest, effective grid, and effective `dt`. The temporal and spatial checks compare five arm means, five relative effects, exact ordering, all target/source numerical gates, and ten materiality/opposite-pair/effort decisions. These safeguards preserve the frozen v1 failure and produced the terminal, round-trip-validated v2 failure above.

### Earlier model portfolio

These configured services belong to the earlier proposal/judge/GEPA design. They do not define the current direct paired-critic screen and are not authorization to invoke it.

| Role | Configured model services |
|---|---|
| Student proposal model | `system.ai.kimi-k3` |
| Primary judge | `system.ai.claude-opus-5` |
| Audit judges | `system.ai.gpt-5-6-sol`, `system.ai.deepseek-v4-pro-0813`, `system.ai.glm-5-2` |
| Reflection models | `system.ai.gpt-5-6-sol`, `system.ai.kimi-k3`, `system.ai.claude-opus-5` |
| Small utility tasks | `system.ai.deepseek-v4-flash-0731` |

Model access uses Databricks' OpenAI-compatible MLflow AI Gateway surface where required. Unity AI Gateway remains an implementation detail rather than the focus of the demo.

### Databricks App

- Built a professional TypeScript/React AppKit replacement for the legacy Streamlit surface.
- The local containment patch is results-only: its server has no job plugin, job trigger route, or launch control, and its shared job-trigger contracts were removed.
- The contained HydroGym evidence surface has:
  - an animated, genuine uncontrolled Kolmogorov vorticity reference trajectory;
  - explicit labeling that the animation is reference physics, not PPO evidence;
  - the Re=100 v2 primary causal pass and terminal temporal/spatial convergence failure shown together, with training still locked;
  - a read-only `critic_quality` review preview filtered to measured evidence with locked fold, bundle/group, arm, and digest tags;
  - separate training-diagnostic and held-out fields;
  - a baseline-versus-approved-candidate chart populated only by dedicated `heldout/*` metrics after physics passes, evidence digests exist, scientific contexts match, frozen PPO fingerprints match, and the candidate carries compiled-reward plus human-approval digests;
  - loading, empty, error, incomplete-provenance, and stale-API-snapshot states;
  - no causal “After MemAlign + GEPA” label.

MLflow trace search exposes request/response previews, not guaranteed full payloads. The contained server therefore returns `REVIEW_WRITE_BLOCKED` for every feedback POST, and the UI disables submission until full native trace retrieval plus RunBundle/evidence-digest verification exists. This intentionally leaves the current expert-label count at zero rather than collecting low-integrity MemAlign targets.

The existing service is still `ACTIVE`, but it is the obsolete deployment and must not be used to launch work. The contained AppKit replacement has not been deployed in this continuation. Deployment requires an explicit review/consent step. The root bundle also includes legacy jobs, so a future rollout must first demonstrate a scoped app-only plan or separate the contained app into its own deployable target; a broad root `apps deploy` must not be assumed safe. Until a contained rollout completes, the live URL does not inherit the local evidence guarantees above.

## Validation and evidence state

Historical app/deployment checks remain useful infrastructure evidence, but they do not make the dirty worktree scientifically valid. Local Python and lint results below validate contracts and plumbing only; the failed Gate 0 remains failed.

| Validation | Result |
|---|---|
| Historical focused Python suite | 91 passed |
| AppKit/Vitest suite | **17 passed** |
| TypeScript checks | Passed |
| ESLint | Passed |
| AppKit AST lint | Passed |
| Prettier check | Passed; generated `appkit.plugins.json` is ignored because AppKit rewrites it |
| Production frontend/server build | Passed |
| Root Databricks bundle validation | Passed |
| Databricks Apps validation | Passed |
| Deployed app status | `ACTIVE`, legacy deployment; contained AppKit patch is not live |
| Authenticated browser render check | Passed |
| Scientific Gate 0 v1 | Durable `c6c1…` development failure records spectral-tail gate failures; exact diagnostic probe measured `0.082028 > 0.05` at `48×48` |
| Scientific Gate 0 v2 | Primary passed all 20 gates; final failed temporal/spatial arm convergence, effect convergence, and ordering; final digest `e8a19f…` |
| Fresh-seed ensemble diagnostic | All causal, numerical, and point-convergence checks passed; both effect-equivalence CIs failed; `supports_designing_full_gate=false`; result digest `e45eb7…` |
| Ten-seed ensemble replication | All 120 H100 trajectories completed; all frozen screens passed; independently audited result digest `c783ea…`; local partial artifact remains quarantined/excluded |
| Control-curl/task-design implementation | Repaired and contract-tested; does not override Gate 0 |
| Direct GPT/Claude transport P0 | Passed one synthetic bundle; remote run `b9908e32d1bb4cb4a633f10530f13bf5` |
| Coding-agent paired sanity | Five groups and ten trace-native records completed; every paired delta was zero; finalizer Job/run `1085457167903195` / `412229067503407` succeeded; HUMAN review session created |
| Bounded real-bug repair pilot | `12/12` exact repairs, `36/36` checks, zero unsafe edits, Wilson lower `0.757506`; Job/run `538885349695793` / `690116555720697`; completed attempt `35953586830762`; MLflow audit `51afc04ff15a43bc85cbb2f8d4776aac` |
| Independent repair-pilot audit | Notebook object `2982479944637503`; Job/run `532806483721171` / `217804453082956` succeeded; both platform attempts independently reproduced direct `12/12`; zero new model/CFD/PPO/MemAlign calls |
| Paired SDK transport P0 | Rejected for MVP: Codex passed, Claude failed closed |
| Native dataset lineage | Local SQLite round-trip passed; publication validates target experiment, successful trace, canonical provenance, RunBundle, root span, and outputs |
| Databricks replication notebook | Uploaded as object `916404733159215`; review run `1052166997412654` succeeded remotely and returned digest-valid structured output with zero CFD |
| Independent replication audit | Job/run `837419045027957` / `383118897658349` succeeded; 120 traces, 240 windows, 840 numerical checks, zero CFD |
| AIR wrapper engineering validation | Notebook object `838420173929111`; run `1112972985772444` passed 15/15 zero-CFD lifecycle and summary-schema checks |
| Gate 0 v3 protocol review | Notebook object `838420173929124`; run `132154788305477` passed 23/23 zero-CFD checks |
| Gate 0 v3 H100 preflight attempt 1 | Run `403428098499749` timed out before the entry point started; no custom MLflow tags/artifacts and zero CFD |
| Gate 0 v3 H100 preflight attempt 2 | Run `199802627560370` succeeded; MLflow `FINISHED`, one H100/JAX x64 confirmed, `gate0_v3/preflight.json` logged, zero CFD |
| Current `test/codex_hydrogym` suite | **252 passed** locally on August 25, 2026 |
| Ruff | Passed on `codex_hydrogym`, its tests, and `hydrogym/jax` |
| Git whitespace check | Passed |
| Bound app jobs | Zero runs on each of the four current jobs |
| Coding-policy H100 PPO execution | Job run `622161538716123` `SUCCESS`; 24 updates, 192 rollouts, ten-file adapter; held-out exact repair `1/12 -> 0/12`, hidden cases `15/36 -> 7/36`, unsafe `0 -> 0`; negative endpoint |

The scoped `test/codex_hydrogym` suite is the relevant project validation boundary. Repository-wide pytest collection also encounters optional Firedrake dependencies and vendored legacy Python-2 tests; those collection failures are outside this demo package and are not counted as project passes.

## Issues encountered and resolutions

### Databricks npm proxy and Playwright

Several deployments failed because the Databricks npm proxy returned `404 Not Found` for `playwright-core` versions 1.62.1, 1.58.2, and 1.58.1 inside the Apps build environment.

Resolution:

- pinned Playwright to a known test version;
- made it an optional, test-only dependency so production installation does not fail when the proxy omits it;
- separated the Playwright e2e TypeScript project from the production client compile;
- retained local smoke-test type checking while removing Playwright from the deployed runtime dependency boundary.

The subsequent deployment installed dependencies, built both server and client, and started successfully.

### Validation-path mismatch

Running `databricks apps validate` from the repository root failed because the root is a bundle and does not directly contain the AppKit `package.json`. Validation was moved to the nested AppKit project, while deployment correctly remained at the repository root to avoid duplicate-app ownership conflicts.

### Python test launcher

The first `uv run pytest` invocation did not place the repository packages on the import path. The suite was rerun through the repository virtual environment with `PYTHONPATH=.`. This exposed one stale assertion for the newly attached PPO job ID; the assertion was corrected, a single-experiment regression test was added, and the full suite passed.

### P0 trace destination mismatch

The P0 calls traced successfully to a local SQLite MLflow store and those traces were serialized into the durable Databricks run. They were not ingested as native remote experiment traces. A read-only `mlflow.search_traces` against experiment `103455306564903` returned zero records, and `mlflow.get_trace` could not resolve the manifest IDs remotely. The result remains valid transport evidence because the manifest and serialized artifacts are present, but it is not a valid source for a remote TRACE-derived labeling dataset. Dataset publication now verifies source existence against the active tracking store so this mismatch fails before labeling.

### Misrouted AI Runtime attempt

An earlier workload targeted the wrong MLflow experiment. That run was canceled, the accidental experiment was moved to Trash, and the reusable AI Runtime job was corrected to use `experiment_name: codex_hydrogym` under `/Workspace/Shared`, resolving to `/Shared/codex_hydrogym`.

### Historical H100 launch failure

Experiment `103455306564903` contains two `FAILED` MLflow runs associated with deleted job `278222053449531` and job run `366806901874578`. Both failed before project user code because the generated `command.sh` attempted to `cd /databricks/code_source/._air`, which was not a directory. Their traces contain only system metrics, including 0% GPU utilization; they contain no PPO, fluid, checkpoint, or physics-gate metrics. This is packaging/setup evidence, not an RL baseline and not evidence of H100 capacity pressure.

### Coding-policy PPO runtime repairs and native-trace limitation

The coding-policy lane encountered three pre-update failures before the terminal run: AppleDouble snapshot-root selection, Databricks CLI 1.9's experimental AIR serializer omitting `code_source_path`, and AIR enabling Hugging Face fast transfer without installing `hf_transfer`. Explicit five-file packaging, raw Jobs API 2.2 updates, direct persistent-Job launch, and pinned `hf-transfer==0.1.9` resolved those failures. Dependency-amended run `1024235606646657` then completed the baseline but exposed a TRL 0.11.4 defect: `RunningMoments` stores float `mean/std` while `PPOTrainer.step()` calls `.to()`. A narrow tensor-type shim preserved the configured score scaling and normalization; completed run `622161538716123` then executed all 24 updates and terminated successfully. None of these repairs used the observed task outcomes to change the corpus, reward, model, seed, or PPO budget.

MLflow artifact upload succeeded and contains the protocol, exact group-disjoint manifest, base/post records, all 24 training rows, isolated snapshots, summary, and ten-file adapter. Native trace-body upload did not: AIR could not connect to the presigned `us-east-1.storage.cloud.databricks.com` trace URLs. The 36 evaluation records contain unique trace IDs, but `mlflow.get_trace()` reports missing span data. The measured artifact result is still auditable, but this run cannot directly seed a trace-native labeling dataset or MemAlign session. No synthetic reconstruction may be represented as HUMAN or native trace evidence.

### Replication backend and wrapper lifecycle

An AIR CPU dry run failed before upload because AIR accepts only GPU accelerator types. After the user explicitly requested GPU capacity when it can reduce runtime, a digest-bound amendment selected one `GPU_1xH100` while retaining JAX float64 and every frozen study input and threshold. The scientific runner completed, but persistent run `767477134906347` was marked failed after the wrapper left its owned MLflow run open and the low-GPU watchdog observed 60 idle minutes. Independent audit established that all decision-bearing artifacts had completed before that wrapper failure. The ownership-aware teardown and self-describing summary fix passed 15 zero-CFD Databricks checks, and real H100 preflight run `784618286017300` plus MLflow run `112972b1ea3f40f3be46d4e7b5442bea` both terminated successfully. This authorization is limited to diagnostics and does not authorize PPO.

### Gate 0 v3 H100 capacity timeout and successful retry

The v3-specific AIR spec dry-run rendered one `GPU_1xH100`, one accelerator, environment version 5, JAX float64 pins, and `expected_trajectory_count=0`. The first submitted preflight, Job/run/task `316596818779054` / `403428098499749` / `108711422687827`, never reached the entry point and timed out after 60 minutes. Its MLflow run `06f57e430b3a435a883861eab9bc23ca` contains only Databricks job tags and no artifacts. The v3 workspace evidence namespace still contains only `protocol.json`. This attempt neither opened reserved cases nor executed CFD and must not be interpreted as a scientific failure or success.

A second idempotent submission retained the same workload and all scientific/digest constraints but raised the outer timeout to 120 minutes. Job/run/task `1057054860106994` / `199802627560370` / `751301736884830` completed `SUCCESS` once capacity arrived; MLflow run `3b16e54b500d49d6a36866d0343ce386` ended `FINISHED` with the expected preflight and protocol artifacts. This completes the H100 readiness check with zero CFD. It does not authorize v3 execution, pass Gate 0, or provide RL evidence.

## Why no reinforcement-learning or alignment results exist yet

This is an evidence gap, not a visualization bug.

1. **The scientific task fails Gate 0 twice.** Re=200 v1 fails the `48×48` spectral-tail criterion. Re=100 v2 passes the primary causal comparison but fails both locked temporal and spatial convergence contracts. Neither authorizes PPO.
2. **The completed H100 code was a fixed-controller diagnostic, not RL.** It executed zero PPO updates, produced no learned policy, and kept `rl_training_performed=false`.
3. **The four currently bound jobs have never run.** Their presence and app permissions prove deployment wiring only.
4. **Transport is not quality.** GPT and Claude direct critics passed a synthetic contract, and the five-group revision sanity pipeline completed, but its paired outcome was neutral and no non-sanity held-out RunBundle or HUMAN quality label has been evaluated.
5. **No expert `critic_quality` labels exist.** MemAlign therefore has neither an attributable training target nor a locked held-out critic-quality evaluation set.
6. **The reward-review revision lane is proven plumbing only.** It consumed five synthetic bundles through the registered prompt and produced trace-native paired outputs, but all quality deltas were zero and no blinded HUMAN comparison shows that its revised reward reasoning is better. Separately, the bounded coding-maintenance proxy passed 12/12 repair-selection cases; that result does not transfer to reward review. GEPA remains outside the first screen.
7. **The existing PPO job targets a different observation problem.** It does not randomize forcing phase per reset, and its training/serving path assumes the legacy speed grid rather than Gate 0's signed observation.
8. **The new causal loop has not executed.** RewardSpec compilation, prompt lineage, AIR manifest enforcement, and frozen-training parity are tested locally, but no human-approved compiled manifest or held-out RL result exists.
9. **The positive claim is narrow.** The coding-model proxy was useful on the frozen bounded maintenance corpus. Agent-feedback quality and fluid-control quality remain separate unproven hypotheses, each requiring its own held-out evidence.

## Truth boundary for the demo

The animated field is a real uncontrolled HydroGym JAX solver reference trajectory. The bounded maintenance pilot shows that the read-only GPT coding-model proxy selected all 12 frozen project repairs correctly under a constrained interface. The active legacy app, contained local AppKit replacement, tests, deployed jobs, direct transport P0, SDK follow-up, registered revision prompt, and measured-local constant-action diagnostic remain foundation or diagnostic evidence. None shows that a learned controller improved the fluid, that coding-agent reward review caused an improvement, or that MemAlign improved feedback.

There are two independent claim boundaries:

- **Reviewer-alignment claim:** one exact Codex draft must feed base-reviewer and MemAlign-reviewer advice, then the same registered prompt and Codex model/adapter must produce both revisions with native remote trace lineage. Experts must label the composite 1–5 `critic_quality` target, and the MemAlign arm must improve held-out human judgment against the base reviewer and cheap deterministic/hand-written controls. MemAlign may improve reward-review feedback. It may not be described as optimizing TKE or control.
- **Fluid-control claim:** the CPU Gate 0 must pass first. A later candidate and baseline must then use identical held-out phases, seeds, developed states, horizons, numerical gates, and effort accounting. Only measured TKE improvement beyond the locked observation-free frontier with proportionate effort supports this claim.

The demo may now say that the Databricks-hosted GPT coding-model proxy selected `12/12` correct bounded repairs on a prospectively locked corpus of actual project incidents, reproduced across both platform attempts; Databricks AI Gateway can host the critic transports; MLflow stores native traces and deterministic audits; fixed signed feedback showed a strong causal TKE effect; and the full held-out v3 gate missed one temporal equivalence boundary by `0.0255923821` percentage points. It cannot say that the model authored or safely applied arbitrary code, that Gate 0 passed, that RL improved anything, that coding-agent reward feedback helped, that Claude review helped, that MemAlign improves the reviewer, or that the system “improves on its own.” A HUMAN must supply the critic-quality target and approve any controller or reward change. GEPA remains optional later work.

## Post-v3 plan

1. Preserve v1, v2, v3, all condition artifacts, raw hashes, and the successful independent-audit output. Do not rewrite the non-round-trippable v3 `result.json`; its defect and exact bytes are evidence.
2. Do not rerun or extend v3, change its `±2%` temporal margin, drop the failed effect pair, reuse its opened cases as unseen, or start fluid-controller PPO.
3. If work continues toward PPO, first perform a prospective design review for a new study ID and untouched sample. Any v4 must address temporal-discretization uncertainty for a physics reason while retaining independently justified margins; it cannot be a threshold relaxation chosen to turn this result into a pass. Execution needs a new explicit authorization.
4. Preserve the bounded maintenance pass as engineering evidence, but do not transfer it to reward review. Coding-agent reward quality and MemAlign reviewer alignment remain separate hypotheses. The five-group reward-review lane exercised the wiring but did not show benefit; MemAlign remains unexecuted and requires attributable grouped HUMAN `critic_quality` labels plus a locked held-out reviewer set.
5. The live App does not update from these workspace JSON files or the audit Job. It remains obsolete until a separately authorized, evidence-only App update/deployment is performed.

## Current handoff

The coding-model proxy now has positive, independently audited evidence on one bounded 12-incident maintenance corpus, and a separate open-model coding-policy PPO run is active. The reward-review architecture remains one Codex harness, a registered revision prompt, HUMAN `critic_quality` labels, base-versus-MemAlign reviewer advice, deterministic human-approved reward compilation, and a coefficient-grid control—but only after valid fluid evidence exists. Its plumbing ran end to end and returned a neutral sanity result; reward-review benefit remains unproven, and MemAlign has not run. Re=100 Gate 0 v3 is a terminal, independently audited failure on one temporal equivalence predicate. Its opened sample cannot be reused and no fluid-controller PPO has run. The next legitimate fluid-science step is prospective design of a distinct gate with untouched cases, not a v3 retry or threshold adjustment.

# codex_hydrogym AI Runtime lane

This workload intentionally starts on one H100. The current JAX PPO learner is
single-device; requesting eight GPUs would reserve hardware it cannot use. Scale the
first experiment as independent one-H100 seeds, then add explicit multi-device JAX
sharding only after profiling proves the simulator is GPU-bound.

The default demo uses an AI Runtime managed environment with pinned JAX and MLflow
dependencies plus a source snapshot. This avoids making Docker Hub credentials a
prerequisite for the first run. The included Dockerfile remains available when an
immutable custom image is useful; custom images cannot be combined with
`environment.version` or `environment.dependencies`.

After selecting an explicit Databricks CLI profile:

```bash
air run --file codex_hydrogym/deploy/air/workload.h100.yaml \
  --dry-run -p <profile>
air run --file codex_hydrogym/deploy/air/workload.h100.yaml \
  --watch -p <profile>
```

Optional custom-image lane:

```bash
docker build --platform linux/amd64 \
  -f codex_hydrogym/deploy/air/Dockerfile \
  -t <dockerhub-org>/codex-hydrogym:0.1.0 .
docker push <dockerhub-org>/codex-hydrogym:0.1.0
air register image <dockerhub-org>/codex-hydrogym:0.1.0 \
  --interactive-authenticate -p <profile>
```

Every run attaches to the MLflow run injected by AI Runtime, logs decomposed physics
and PPO metrics, evaluates hard numerical gates, and uploads a checksummed restart
checkpoint. A passing run also logs a deterministic MLflow pyfunc, registers the same
artifact in Unity Catalog, and moves only the `candidate` alias. The separate guarded
promotion job controls `production` using comparable held-out PPO evidence.

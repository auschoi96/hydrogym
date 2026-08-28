#!/usr/bin/env bash
set -euo pipefail

export CODEX_HYDROGYM_ACTION=run
export CODEX_HYDROGYM_OUTPUT_DIR=/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/evidence/269507101a52-a5ab894e5ff4/databricks-primary-20260825
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$CODE_SOURCE_PATH"
python -m pip install --disable-pip-version-check --no-deps notebooks/artifacts/hydrogym-1.0.0-py3-none-any.whl
python deploy/air/gate0_replication_entrypoint.py


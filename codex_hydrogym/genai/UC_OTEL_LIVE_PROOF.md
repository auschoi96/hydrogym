# Unity Catalog OTel live-delivery proof

This runbook sends **one synthetic transport probe** through the existing
`analyze_run_bundle` harness tracing path. It is not a coding-agent evaluation,
model call, or fluid-control result. The probe writes only to the throwaway
MLflow experiment `/Shared/codex_hydrogym_uctrace_probe_v3` and to tables with
the prefix `codex_hydrogym_uctrace_probe_20260829_v3`. Its root-span attributes
include `bundle_id=uc_otel_throwaway_probe_20260829`, and its artifacts use the
`synthetic://` scheme. Those markers distinguish it from attested evidence.

## Reproduce

MLflow 3.14+ is required because experiment-bound Unity Catalog trace locations
create the four OTel tables. Do not change the repository virtual environment;
use a temporary one if needed.

```bash
cd /path/to/hydrogym
python3 -m venv --system-site-packages /tmp/hydrogym-uc-mlflow-venv
/tmp/hydrogym-uc-mlflow-venv/bin/pip install 'mlflow[databricks]>=3.14,<4'

# Keep DATABRICKS_CONFIG_PROFILE unset. Obtain a short-lived token by passing
# the selected profile explicitly to the CLI, then pass that token only to this process.
unset DATABRICKS_CONFIG_PROFILE
DATABRICKS_TOKEN="$(databricks auth token --profile dais-demo | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')" \
DATABRICKS_HOST=https://fevm-austin-choi-omni-agent.cloud.databricks.com \
CODEX_HYDROGYM_ENABLE_UC_TRACING=1 \
MLFLOW_TRACING_UC_CATALOG_NAME=austin_choi_omni_agent_catalog \
MLFLOW_TRACING_UC_SCHEMA_NAME=codex_hydrogym \
MLFLOW_TRACING_SQL_WAREHOUSE_ID=425a17963117ad03 \
PYTHONPATH=. /tmp/hydrogym-uc-mlflow-venv/bin/python -m codex_hydrogym.genai.uc_trace_probe --profile dais-demo
```

The probe prints its `trace_id`. Query it using the same explicit profile and
warehouse (the warehouse autostarts if stopped):

```bash
unset DATABRICKS_CONFIG_PROFILE
databricks experimental aitools tools query --profile dais-demo --warehouse 425a17963117ad03 --output json \
  "SELECT count(*) AS span_rows FROM austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_uctrace_probe_20260829_v3_otel_spans" \
  "SELECT trace_id, span_id, name FROM austin_choi_omni_agent_catalog.codex_hydrogym.codex_hydrogym_uctrace_probe_20260829_v3_otel_spans WHERE trace_id = '<trace_id printed above>' ORDER BY start_time_unix_nano"
```

## Live result (2026-08-29)

The run used profile `dais-demo`, warehouse `425a17963117ad03`, catalog
`austin_choi_omni_agent_catalog`, schema `codex_hydrogym`, and the opt-in flag.
It emitted trace ID `118630e8a5359ae4e51adb2c27748b8d`. The UC query returned
`span_rows = 6` and the same trace ID for these harness spans:

```text
hydrogym_feedback_agent
harness_call
contract_validation
```

The experiment-bound location created these throwaway OTel tables:

```text
codex_hydrogym_uctrace_probe_20260829_v3_otel_annotations
codex_hydrogym_uctrace_probe_20260829_v3_otel_logs
codex_hydrogym_uctrace_probe_20260829_v3_otel_metrics
codex_hydrogym_uctrace_probe_20260829_v3_otel_spans
```

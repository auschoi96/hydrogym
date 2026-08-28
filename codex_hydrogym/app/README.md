# codex_hydrogym Databricks App

Single-process Streamlit delivery surface for the fresh demo deployment.

- App service-principal OAuth performs shared MLflow writes and Unity AI Gateway calls.
- Forwarded Databricks user identity attributes every human assessment.
- MLflow is the system of record; the App does not read Unity Catalog tables directly.
- `app.yaml` model placeholders must be replaced only after workspace model-service discovery.
- The MLflow experiment is attached as the App resource key `experiment` with edit access.

Local smoke test:

```bash
CODEX_HYDROGYM_REVIEWER=local@example.com \
  streamlit run codex_hydrogym/app/app.py
```


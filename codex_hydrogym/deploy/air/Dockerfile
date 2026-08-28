FROM databricksruntime/air:dcs-base-aws-runtime

LABEL org.opencontainers.image.title="codex_hydrogym"
LABEL org.opencontainers.image.description="HydroGym JAX PPO training for Databricks AI Runtime"
LABEL com.databricks.demo.project="codex_hydrogym"

COPY pyproject.toml README.md /app/hydrogym/
COPY hydrogym /app/hydrogym/hydrogym
COPY codex_hydrogym /app/hydrogym/codex_hydrogym

RUN uv pip install --no-cache \
    "/app/hydrogym[jax,codex_hydrogym]" \
    "jax[cuda12]==0.7.2" \
    "jaxlib==0.7.2" \
    "flax==0.12.0" \
    "optax==0.2.8" \
    "distrax==0.1.9" \
    "gymnax==0.0.9" \
    "chex==0.1.92" \
    "navix==0.11.0" \
    "tree-math==0.2.1" \
    "mlflow==3.11.1"

WORKDIR /app/hydrogym

CMD ["python", "-m", "codex_hydrogym.training.air_entrypoint"]

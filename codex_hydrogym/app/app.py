"""Streamlit review console for the codex_hydrogym Databricks demo."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os

import streamlit as st

try:
    from codex_hydrogym.app.backend import (
        AppModelPortfolio,
        UnityGateway,
        recent_training_runs,
        resolve_app_token,
        reviewer_identity,
        search_review_traces,
        submit_human_feedback,
    )
except ModuleNotFoundError:  # Databricks deploys this directory as the App source root.
    from backend import (
        AppModelPortfolio,
        UnityGateway,
        recent_training_runs,
        resolve_app_token,
        reviewer_identity,
        search_review_traces,
        submit_human_feedback,
    )


st.set_page_config(page_title="codex_hydrogym · Turbulence in the Loop", page_icon="🌊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1440px; padding-top: 1.4rem;}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(42, 151, 177, .35);
             border-radius: 16px; background: linear-gradient(120deg, rgba(10,42,58,.96), rgba(9,73,83,.88));
             color: #f3fbfd; margin-bottom: 1rem;}
      .hero h1 {margin: 0; font-size: 2.05rem;}
      .hero p {margin: .35rem 0 0; color: #bfe8ef;}
      .signal {padding: .75rem 1rem; border-radius: 12px; background: rgba(37, 150, 190, .08);
               border: 1px solid rgba(37, 150, 190, .22);}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
    </style>
    <div class="hero">
      <h1>codex_hydrogym · Turbulence in the Loop</h1>
      <p>Human-aligned reward discovery for JAX PPO fluid control on Databricks AI Runtime.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _headers():
    try:
        return st.context.headers
    except Exception:
        return {}


def _configure_mlflow():
    import mlflow

    mlflow.set_tracking_uri("databricks")
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "").strip()
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)
    return mlflow, experiment_id


@st.cache_resource(ttl=300)
def _gateway():
    host = os.environ.get("DATABRICKS_HOST", "")
    return UnityGateway(workspace_host=host, token=resolve_app_token())


def _pretty(value):
    if value is None:
        return "No payload recorded"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


mlflow, experiment_id = _configure_mlflow()
reviewer = reviewer_identity(_headers())

with st.sidebar:
    st.subheader("Demo status")
    st.caption("Project")
    st.code("codex_hydrogym", language=None)
    st.caption("MLflow experiment")
    st.code(experiment_id or "not attached", language=None)
    st.caption("Reviewer")
    st.code(reviewer or "identity unavailable", language=None)
    if experiment_id and reviewer:
        st.success("Ready for attributable feedback")
    else:
        st.warning("Review submission is disabled until experiment and identity are available")
    st.divider()
    st.caption(
        "Research demo · AI-generated proposals can be wrong. Physics gates and human review remain authoritative."
    )

review_tab, model_tab, evidence_tab, learn_tab = st.tabs(
    ["Review & label", "Model lab", "PPO evidence", "How it learns"]
)

with review_tab:
    st.subheader("Domain-expert feedback")
    st.write(
        "Score the same `fluid_reward_plausibility` criterion used by the primary MLflow judge. "
        "Comments become MemAlign memory."
    )
    if not experiment_id:
        st.info("Attach the `codex_hydrogym` MLflow experiment resource to load review traces.")
    else:
        try:
            traces = search_review_traces(experiment_id=experiment_id, mlflow_module=mlflow)
        except Exception as error:
            st.error(f"Could not load MLflow review traces: {type(error).__name__}: {error}")
            traces = []
        if not traces:
            st.info("No review-ready traces yet. Run the codex_hydrogym bootstrap job first.")
        else:
            pending = sum(trace.review_state == "pending_human" for trace in traces)
            labels = sum(
                assessment["name"] == "fluid_reward_plausibility" and assessment["source_type"] == "HUMAN"
                for trace in traces
                for assessment in trace.assessments
            )
            a, b, c = st.columns(3)
            a.metric("Review-ready traces", len(traces))
            b.metric("Pending", pending)
            c.metric("Human labels", labels)
            trace_by_id = {trace.trace_id: trace for trace in traces}
            selected_id = st.selectbox("Trace", list(trace_by_id), format_func=lambda value: value[:18] + "…")
            selected = trace_by_id[selected_id]
            left, right = st.columns(2)
            with left:
                st.markdown("**Scenario / judge input**")
                st.code(_pretty(selected.request), language="json")
            with right:
                st.markdown("**Reward proposal / rollout evidence**")
                st.code(_pretty(selected.response), language="json")
            human_assessments = [
                assessment
                for assessment in selected.assessments
                if assessment["name"] == "fluid_reward_plausibility" and assessment["source_type"] == "HUMAN"
            ]
            if human_assessments:
                st.caption(f"{len(human_assessments)} human assessment(s) already attached to this trace.")
            with st.form(f"feedback-{selected.trace_id}", clear_on_submit=True):
                score = st.slider("Fluid reward plausibility", 1, 5, 3)
                rationale = st.text_area(
                    "Why?",
                    placeholder="Explain the decisive physics, control-effort, or validation concern.",
                    max_chars=4_000,
                )
                submitted = st.form_submit_button("Submit attributable MLflow feedback", disabled=not reviewer)
                if submitted:
                    try:
                        submit_human_feedback(
                            trace_id=selected.trace_id,
                            score=score,
                            rationale=rationale,
                            reviewer=reviewer,
                            mlflow_module=mlflow,
                        )
                    except Exception as error:
                        st.error(f"Feedback was not saved: {type(error).__name__}: {error}")
                    else:
                        st.success("Feedback saved to the MLflow trace for MemAlign.")

with model_tab:
    st.subheader("Direct Unity AI Gateway model lab")
    st.write("Calls use the workspace OpenAI-compatible `/ai-gateway/mlflow/v1` surface and App OAuth.")
    try:
        portfolio = AppModelPortfolio.from_env()
    except ValueError as error:
        st.warning(str(error))
        portfolio = None
    if portfolio:
        roles = portfolio.role_models()
        role = st.selectbox("Model role", list(roles))
        model = roles[role]
        st.caption(f"Verified model service: `{model}`")
        with st.form("model-lab"):
            prompt = st.text_area(
                "Prompt",
                value=(
                    "Critique this reward hypothesis for a Re=200 Kolmogorov flow while preserving hard physics gates."
                ),
                max_chars=12_000,
            )
            invoke = st.form_submit_button("Call Unity AI Gateway")
        if invoke:
            try:
                with st.spinner(f"Calling {model}…"):
                    response = _gateway().chat(model=model, prompt=prompt)
            except Exception as error:
                st.error(f"Gateway call failed: {type(error).__name__}: {error}")
            else:
                st.markdown(response["text"])
                st.caption(
                    f"model={response['model']} · request_id={response['request_id']} · "
                    f"usage={json.dumps(response['usage'], sort_keys=True)}"
                )

with evidence_tab:
    st.subheader("Held-out PPO evidence")
    st.write("Production promotion requires comparable context, lower mean TKE, bounded control effort, and all gates.")
    if not experiment_id:
        st.info("Attach the MLflow experiment to load H100 PPO runs.")
    else:
        try:
            runs = recent_training_runs(experiment_id=experiment_id, mlflow_module=mlflow)
        except Exception as error:
            st.error(f"Could not load PPO runs: {type(error).__name__}: {error}")
            runs = []
        if runs:
            st.dataframe(runs, use_container_width=True, hide_index=True)
        else:
            st.info("No labeled JAX PPO runs are available yet.")

with learn_tab:
    st.subheader("Two-speed learning loop")
    st.markdown(
        """
        <div class="signal"><b>Fast loop</b> · JAX PPO on H100 → deterministic CFD gates →
        MLflow metrics and artifacts</div>
        <p style="text-align:center; font-size:1.4rem; margin:.45rem">↕</p>
        <div class="signal"><b>Slow loop</b> · student proposals → judge portfolio → human labels →
        MemAlign → GEPA</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        - LLMs may propose only bounded scalar candidates; they cannot emit reward code or mutate the solver.
        - The primary judge and label schema share the exact name `fluid_reward_plausibility`.
        - MemAlign uses human comments plus `databricks:/databricks-gte-large-en` retrieval.
        - GEPA may create only a `candidate` prompt alias. Held-out PPO gates control `production`.
        """
    )
    st.caption(f"Rendered {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

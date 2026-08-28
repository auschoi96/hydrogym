"""MLflow lifecycle contracts for codex_hydrogym."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig
from codex_hydrogym.tracking import evaluation_context_fingerprint, managed_mlflow_run


@dataclass
class _RunInfo:
    run_id: str


@dataclass
class _Run:
    info: _RunInfo


class _FakeMlflow:
    def __init__(self, active_run=None):
        self._active_run = active_run
        self.start_calls = []
        self.end_calls = []
        self.tags = {}

    def active_run(self):
        return self._active_run

    def start_run(self, **kwargs):
        self.start_calls.append(kwargs)
        run_id = kwargs.get("run_id", "new-run")
        self._active_run = _Run(_RunInfo(run_id))
        return self._active_run

    def set_tags(self, tags):
        self.tags.update(tags)

    def end_run(self, status="FINISHED"):
        self.end_calls.append(status)
        self._active_run = None


def test_air_run_id_is_attached_and_labeled(monkeypatch):
    monkeypatch.setenv("MLFLOW_RUN_ID", "air-run-123")
    mlflow = _FakeMlflow()

    with managed_mlflow_run(mlflow_module=mlflow, component="ppo") as run:
        assert run.info.run_id == "air-run-123"

    assert mlflow.start_calls == [{"run_id": "air-run-123"}]
    assert mlflow.end_calls == ["FINISHED"]
    assert mlflow.tags["codex_hydrogym.project"] == PROJECT_LABEL
    assert mlflow.tags["codex_hydrogym.component"] == "ppo"
    assert mlflow.tags["codex_hydrogym.run_origin"] == "ai_runtime"


def test_existing_active_run_is_reused_without_being_closed(monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    existing = _Run(_RunInfo("existing-run"))
    mlflow = _FakeMlflow(active_run=existing)

    with managed_mlflow_run(mlflow_module=mlflow, component="gepa") as run:
        assert run is existing

    assert mlflow.start_calls == []
    assert mlflow.end_calls == []
    assert mlflow.tags["codex_hydrogym.component"] == "gepa"


def test_new_standalone_run_name_is_prefixed(monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow = _FakeMlflow()

    with managed_mlflow_run(mlflow_module=mlflow, component="evaluation", run_name="smoke"):
        pass

    assert mlflow.start_calls == [{"run_name": "codex_hydrogym_smoke"}]
    assert mlflow.tags["codex_hydrogym.run_origin"] == "standalone"


def test_owned_run_is_marked_failed_on_exception(monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    mlflow = _FakeMlflow()

    with pytest.raises(RuntimeError, match="boom"):
        with managed_mlflow_run(mlflow_module=mlflow, component="ppo"):
            raise RuntimeError("boom")

    assert mlflow.end_calls == ["FAILED"]


def test_evaluation_context_excludes_candidate_tunables_but_includes_budget_and_mechanics():
    baseline = KolmogorovPPOConfig()
    candidate = KolmogorovPPOConfig(
        reward_alpha=2.0,
        learning_rate=2.0e-4,
        entropy_coefficient=0.01,
        gamma=0.995,
        gae_lambda=0.98,
    )
    larger_budget = KolmogorovPPOConfig(total_timesteps=8_192)
    shifted_phase = KolmogorovPPOConfig(forcing_phase=0.5)
    signed_observations = KolmogorovPPOConfig(observation_mode="signed_forced_mode")
    alternate_basis = SimpleNamespace(**vars(baseline))
    alternate_basis.actuation_basis_version = "future_incompatible_basis"

    assert evaluation_context_fingerprint(baseline) == evaluation_context_fingerprint(candidate)
    assert evaluation_context_fingerprint(baseline) != evaluation_context_fingerprint(larger_budget)
    assert evaluation_context_fingerprint(baseline) != evaluation_context_fingerprint(shifted_phase)
    assert evaluation_context_fingerprint(baseline) != evaluation_context_fingerprint(signed_observations)
    assert evaluation_context_fingerprint(baseline) != evaluation_context_fingerprint(alternate_basis)

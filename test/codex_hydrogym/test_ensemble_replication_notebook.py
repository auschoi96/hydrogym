"""Safety and provenance contracts for the Databricks replication notebook."""

import ast
import json
import os
from pathlib import Path
import runpy


NOTEBOOK = Path(__file__).resolve().parents[2] / "codex_hydrogym" / "notebooks" / "ensemble_replication.py"


def test_notebook_is_source_format_and_syntax_valid():
    source = NOTEBOOK.read_text(encoding="utf-8")

    assert source.startswith("# Databricks notebook source\n")
    ast.parse(source)


def test_notebook_defaults_to_review_and_requires_exact_run_confirmation():
    source = NOTEBOOK.read_text(encoding="utf-8")

    assert '_ensure_widget("action", "review", ["review", "install", "preflight", "run"])' in source
    assert 'if ACTION == "run":' in source
    assert "if CONFIRMATION != CONFIRMATION_TOKEN:" in source
    assert "RUN_PRIMARY_DATABRICKS_REPLICATION:" in source
    assert '"claim_role": "primary_decision_bearing_execution"' in source
    assert '"decision_bearing_execution": True' in source
    assert '"sole_analysis_set": True' in source
    assert '"local_partial_artifact_in_analysis": False' in source
    assert '"prior_or_local_results_in_analysis": 0' in source


def test_notebook_binds_the_frozen_protocol_source_and_environment():
    source = NOTEBOOK.read_text(encoding="utf-8")

    assert "269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62" in source
    assert "a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c" in source
    assert "3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87" in source
    assert "91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1" in source
    assert 'os.environ["JAX_ENABLE_X64"] = "1"' in source
    assert '"jax": "0.7.2"' in source
    assert 'protocol["implementation_files"]' in source


def test_notebook_does_not_bind_a_workspace_profile_or_root_bundle():
    source = NOTEBOOK.read_text(encoding="utf-8")

    assert "--profile" not in source
    assert "dais-demo" not in source
    assert "bundle deploy" not in source


def test_notebook_review_mode_round_trips_the_companion_wheel(tmp_path):
    wheel = NOTEBOOK.parent / "artifacts" / "hydrogym-1.0.0-py3-none-any.whl"

    class _Widgets:
        values = {
            "action": "review",
            "confirmation": "",
            "output_dir": str(tmp_path / "unused"),
            "wheel_path": str(wheel),
        }

        def get(self, name):
            return self.values[name]

        def text(self, name, default):
            self.values[name] = default

        def dropdown(self, name, default, _choices):
            self.values[name] = default

    class _Notebook:
        exit_payloads = []

        def exit(self, payload):
            self.exit_payloads.append(payload)

    class _Dbutils:
        widgets = _Widgets()
        notebook = _Notebook()

    # The notebook pins JAX env vars for the Databricks GPU runtime; keep that
    # process-global mutation from leaking into the rest of the test session.
    jax_env = {name: os.environ.get(name) for name in ("JAX_ENABLE_X64", "JAX_PLATFORMS")}
    try:
        namespace = runpy.run_path(str(NOTEBOOK), init_globals={"dbutils": _Dbutils()})
    finally:
        for name, value in jax_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert namespace["WHEEL_REVIEW"]["artifact_and_source_hashes_valid"] is True
    assert namespace["WHEEL_REVIEW"]["implementation_file_count"] == 8
    assert namespace["RUN_SUMMARY"] is None
    assert len(_Notebook.exit_payloads) == 1
    review_payload = json.loads(_Notebook.exit_payloads[0])
    assert review_payload["action"] == "review"
    assert review_payload["cfds_executed"] == 0

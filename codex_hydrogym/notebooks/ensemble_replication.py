# Databricks notebook source
# ruff: noqa: E402, F821

# COMMAND ----------

# MAGIC %md
# MAGIC # HydroGym Re=100 ten-seed ensemble replication
# MAGIC
# MAGIC This notebook packages the exact frozen, development-only replication study for review and
# MAGIC its primary execution on Databricks AI Runtime with one H100.
# MAGIC
# MAGIC **Scientific boundary:** the local process was stopped during a user-directed platform switch
# MAGIC after one base-condition artifact had completed. That artifact was quarantined without opening
# MAGIC its metrics and is excluded from analysis. The full Databricks run from this notebook is the
# MAGIC **sole decision-bearing execution**. Do not pool the local partial artifact, use it as an interim
# MAGIC look, replace seeds, extend the sample, reinterpret Gate 0, or authorize PPO.
# MAGIC
# MAGIC The default `action=review` cannot install packages or execute CFD. The managed Job runs a
# MAGIC non-CFD `action=preflight` task before `action=run`. To run manually:
# MAGIC
# MAGIC 1. Use one H100 with Python 3.12.
# MAGIC 2. Upload the reviewed wheel, `platform_transition.json`, and
# MAGIC    `execution_backend_amendment.json` beside this notebook.
# MAGIC 3. Select `action=install`, run this notebook once, and allow Python to restart.
# MAGIC 4. Select `action=preflight` to validate Python, H100 JAX, x64, package pins, and source hashes.
# MAGIC 5. Select `action=run`, enter the exact confirmation token shown below, and run again.
# MAGIC 6. Use the frozen durable `/Workspace/Users/...` output path. The study resumes only at
# MAGIC    completed condition boundaries.
# MAGIC
# MAGIC This is serial, driver-only JAX work. It does not use Spark workers. The original CPU route was
# MAGIC amended, blind to partial metrics, after explicit user direction to use GPU capacity when faster.
# MAGIC Float64 remains mandatory and the backend check fails closed unless the runtime is an H100 GPU.

# COMMAND ----------

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile


STUDY_ID = "re100_fresh_seed_windowed_convergence_replication_v1"
STUDY_FINGERPRINT = "269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62"
IMPLEMENTATION_DIGEST = "a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c"
PROTOCOL_ARTIFACT_DIGEST = "3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87"
WHEEL_FILENAME = "hydrogym-1.0.0-py3-none-any.whl"
WHEEL_SHA256 = "91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1"
TRANSITION_FILENAME = "platform_transition.json"
TRANSITION_SHA256 = "69f000608ae8b0aa9b0d8f3433ce1108617936e20f9763d77fe398ad5c96a428"
TRANSITION_ARTIFACT_DIGEST = "deb256b550dd7d3d0fc88746db4ca7b0cbcdeb60f8b921d4fe19bb0466ad2e8a"
BACKEND_AMENDMENT_FILENAME = "execution_backend_amendment.json"
BACKEND_AMENDMENT_SHA256 = "ec39d51fdbe288080a52730f5d665acea4ee8bcb1ee00c26f7900a2107156bdf"
BACKEND_AMENDMENT_ARTIFACT_DIGEST = (
    "28747c56c53d2dd251dee8f17f49ef1c67c5d5b01d5fc9b3ea9d1b8f8c84e181"
)
PACKAGED_PROTOCOL = "codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4/protocol.json"
PRIMARY_OUTPUT_DIR = (
    "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_replication/"
    "evidence/269507101a52-a5ab894e5ff4/databricks-primary-20260825"
)
CONFIRMATION_TOKEN = f"RUN_PRIMARY_DATABRICKS_REPLICATION:{STUDY_FINGERPRINT}"
PINNED_PACKAGES = {
    "chex": "0.1.92",
    "flax": "0.12.0",
    "gymnasium": "1.3.0",
    "gymnax": "0.0.9",
    "huggingface-hub": "1.27.0",
    "jax": "0.7.2",
    "jaxlib": "0.7.2",
    "matplotlib": "3.11.1",
    "navix": "0.11.0",
    "numpy": "2.5.2",
    "omegaconf": "2.3.1",
    "scipy": "1.18.0",
    "toml": "0.10.2",
    "tree-math": "0.2.1",
}


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _notebook_directory() -> Path:
    try:
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        notebook_path = str(context.notebookPath().get())
        return Path("/Workspace") / Path(notebook_path.lstrip("/")).parent
    except Exception:
        return Path.cwd()


def _ensure_widget(name: str, default: str, choices: list[str] | None = None) -> str:
    try:
        return str(dbutils.widgets.get(name))
    except Exception:
        if choices is None:
            dbutils.widgets.text(name, default)
        else:
            dbutils.widgets.dropdown(name, default, choices)
        return str(dbutils.widgets.get(name))


NOTEBOOK_DIRECTORY = _notebook_directory()
ACTION = _ensure_widget("action", "review", ["review", "install", "preflight", "run"])
WHEEL_PATH = _ensure_widget("wheel_path", str(NOTEBOOK_DIRECTORY / WHEEL_FILENAME))
TRANSITION_PATH = _ensure_widget("transition_path", str(NOTEBOOK_DIRECTORY / TRANSITION_FILENAME))
BACKEND_AMENDMENT_PATH = _ensure_widget(
    "backend_amendment_path", str(NOTEBOOK_DIRECTORY / BACKEND_AMENDMENT_FILENAME)
)
OUTPUT_DIR = _ensure_widget("output_dir", PRIMARY_OUTPUT_DIR)
CONFIRMATION = _ensure_widget("confirmation", "")

# This must be set before importing JAX in a fresh Python process.
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cuda"

print(
    json.dumps(
        {
            "action": ACTION,
            "backend_amendment_path": BACKEND_AMENDMENT_PATH,
            "claim_role": "primary_decision_bearing_execution",
            "confirmation_token": CONFIRMATION_TOKEN,
            "expected_trajectory_count": 120,
            "implementation_digest": IMPLEMENTATION_DIGEST,
            "output_dir": OUTPUT_DIR,
            "study_fingerprint": STUDY_FINGERPRINT,
            "transition_path": TRANSITION_PATH,
            "wheel_path": WHEEL_PATH,
        },
        indent=2,
        sort_keys=True,
    )
)

# COMMAND ----------


def _review_wheel(*, required: bool) -> dict[str, object] | None:
    wheel_path = Path(WHEEL_PATH)
    if not wheel_path.is_file():
        message = f"Upload {WHEEL_FILENAME} to {wheel_path} before install/run."
        if required:
            raise RuntimeError(message)
        print(message)
        return None

    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if wheel_digest != WHEEL_SHA256:
        raise RuntimeError(f"wheel SHA-256 mismatch: expected {WHEEL_SHA256}, observed {wheel_digest}")
    with zipfile.ZipFile(wheel_path) as archive:
        protocol = json.loads(archive.read(PACKAGED_PROTOCOL))
        artifact_digest = protocol.get("artifact_digest")
        body = {key: value for key, value in protocol.items() if key != "artifact_digest"}
        if artifact_digest != _digest(body) or artifact_digest != PROTOCOL_ARTIFACT_DIGEST:
            raise RuntimeError("packaged protocol artifact digest validation failed")
        if protocol.get("study_fingerprint") != STUDY_FINGERPRINT:
            raise RuntimeError("packaged protocol study fingerprint mismatch")
        if protocol.get("implementation_digest") != IMPLEMENTATION_DIGEST:
            raise RuntimeError("packaged protocol implementation digest mismatch")
        for relative, expected_digest in protocol["implementation_files"].items():
            observed_digest = hashlib.sha256(archive.read(relative)).hexdigest()
            if observed_digest != expected_digest:
                raise RuntimeError(f"packaged source hash mismatch: {relative}")

    summary = {
        "artifact_and_source_hashes_valid": True,
        "fixed_seed_count": len(protocol["spec"]["seeds"]),
        "implementation_file_count": len(protocol["implementation_files"]),
        "protocol_artifact_digest": artifact_digest,
        "reserved_phase_turns": protocol["spec"]["reserved_phase_turns"],
        "reserved_seeds": protocol["spec"]["reserved_seeds"],
        "seeds": protocol["spec"]["seeds"],
        "wheel_sha256": wheel_digest,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _review_transition(*, required: bool) -> dict[str, object] | None:
    transition_path = Path(TRANSITION_PATH)
    if not transition_path.is_file():
        message = f"Upload {TRANSITION_FILENAME} to {transition_path} before preflight/run."
        if required:
            raise RuntimeError(message)
        print(message)
        return None

    raw_digest = hashlib.sha256(transition_path.read_bytes()).hexdigest()
    if raw_digest != TRANSITION_SHA256:
        raise RuntimeError(
            f"transition manifest SHA-256 mismatch: expected {TRANSITION_SHA256}, observed {raw_digest}"
        )
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    artifact_digest = transition.get("artifact_digest")
    body = {key: value for key, value in transition.items() if key != "artifact_digest"}
    if artifact_digest != _digest(body) or artifact_digest != TRANSITION_ARTIFACT_DIGEST:
        raise RuntimeError("platform-transition artifact digest validation failed")
    if transition.get("study_fingerprint") != STUDY_FINGERPRINT:
        raise RuntimeError("platform-transition study fingerprint mismatch")
    if transition["local_execution"]["partial_artifact"]["content_or_metrics_inspected"] is not False:
        raise RuntimeError("local partial artifact is not recorded as blind")
    databricks_execution = transition["databricks_execution"]
    if databricks_execution["output_namespace"] != OUTPUT_DIR:
        raise RuntimeError("output_dir differs from the frozen Databricks output namespace")
    if databricks_execution["prior_or_local_results_in_analysis"] != 0:
        raise RuntimeError("platform transition does not exclude prior/local results")
    if databricks_execution["sole_analysis_set"] is not True:
        raise RuntimeError("Databricks execution is not recorded as the sole analysis set")

    summary = {
        "artifact_digest": artifact_digest,
        "decision_made_blind_to_local_partial_metrics": True,
        "local_partial_analysis_eligibility": transition["local_execution"]["analysis_eligibility"],
        "output_namespace": databricks_execution["output_namespace"],
        "raw_sha256": raw_digest,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _review_backend_amendment(*, required: bool) -> dict[str, object] | None:
    amendment_path = Path(BACKEND_AMENDMENT_PATH)
    if not amendment_path.is_file():
        message = f"Upload {BACKEND_AMENDMENT_FILENAME} to {amendment_path} before preflight/run."
        if required:
            raise RuntimeError(message)
        print(message)
        return None

    raw_digest = hashlib.sha256(amendment_path.read_bytes()).hexdigest()
    if raw_digest != BACKEND_AMENDMENT_SHA256:
        raise RuntimeError(
            "backend-amendment SHA-256 mismatch: "
            f"expected {BACKEND_AMENDMENT_SHA256}, observed {raw_digest}"
        )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    artifact_digest = amendment.get("artifact_digest")
    body = {key: value for key, value in amendment.items() if key != "artifact_digest"}
    if artifact_digest != _digest(body) or artifact_digest != BACKEND_AMENDMENT_ARTIFACT_DIGEST:
        raise RuntimeError("execution-backend amendment artifact digest validation failed")
    if amendment.get("study_fingerprint") != STUDY_FINGERPRINT:
        raise RuntimeError("execution-backend amendment study fingerprint mismatch")
    if amendment.get("previous_transition_artifact_digest") != TRANSITION_ARTIFACT_DIGEST:
        raise RuntimeError("execution-backend amendment does not bind the platform transition")
    backend = amendment["authorized_change"]
    expected_backend = {
        "accelerator_count": 1,
        "accelerator_type": "GPU_1xH100",
        "execution_service": "Databricks AI Runtime",
        "jax_backend": "gpu",
        "precision": "float64",
    }
    if backend != expected_backend:
        raise RuntimeError("execution-backend amendment differs from the reviewed H100 contract")
    safeguards = amendment["scientific_safeguards"]
    if safeguards["protocol_seeds_or_thresholds_changed"] is not False:
        raise RuntimeError("backend amendment changed the frozen study design")
    if safeguards["local_partial_artifact_in_analysis"] is not False:
        raise RuntimeError("backend amendment does not exclude the local partial artifact")

    summary = {
        "artifact_digest": artifact_digest,
        "authorized_change": backend,
        "raw_sha256": raw_digest,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


WHEEL_REVIEW = _review_wheel(required=ACTION in {"install", "preflight", "run"})
TRANSITION_REVIEW = _review_transition(required=ACTION in {"preflight", "run"})
BACKEND_AMENDMENT_REVIEW = _review_backend_amendment(required=ACTION in {"preflight", "run"})
if ACTION == "review":
    print("Review complete. No package installation or CFD execution was performed.")

# COMMAND ----------


if ACTION == "install":
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"the reviewed environment requires Python 3.12, observed {sys.version.split()[0]}")
    import subprocess

    requirements = [f"{name}=={version}" for name, version in PINNED_PACKAGES.items()]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirements])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            WHEEL_PATH,
        ]
    )
    print("Pinned dependencies and the reviewed wheel are installed. Restarting Python.")
    print("After restart, select action=run and enter the exact confirmation token.")
    dbutils.library.restartPython()
elif ACTION == "review":
    print("Install step skipped because action=review.")

# COMMAND ----------


def _validate_runtime_environment() -> dict[str, object]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"the reviewed environment requires Python 3.12, observed {sys.version.split()[0]}")

    from importlib.metadata import version as package_version

    observed_packages = {name: package_version(name) for name in PINNED_PACKAGES}
    if observed_packages != PINNED_PACKAGES:
        raise RuntimeError(
            "installed package versions differ from the reviewed environment: "
            + json.dumps(observed_packages, sort_keys=True)
        )

    import jax

    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX float64 could not be enabled")
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("the amended Databricks execution requires GPU-backed JAX")
    if any("H100" not in device.device_kind.upper() for device in jax.devices()):
        raise RuntimeError("the amended Databricks execution requires an H100 device")

    from codex_hydrogym.gate0 import ensemble_replication

    spec = ensemble_replication.EnsembleReplicationSpec()
    implementation_files, implementation_digest = ensemble_replication._study_implementation_manifest()
    if spec.fingerprint != STUDY_FINGERPRINT:
        raise RuntimeError("installed replication spec fingerprint mismatch")
    if implementation_digest != IMPLEMENTATION_DIGEST:
        raise RuntimeError("installed replication implementation digest mismatch")

    packaged_protocol_path = (
        Path(ensemble_replication.__file__).resolve().parents[1]
        / "evidence"
        / "ensemble_replication"
        / "269507101a52-a5ab894e5ff4"
        / "protocol.json"
    )
    packaged_protocol = json.loads(packaged_protocol_path.read_text(encoding="utf-8"))
    if packaged_protocol["artifact_digest"] != PROTOCOL_ARTIFACT_DIGEST:
        raise RuntimeError("installed protocol artifact identity mismatch")
    if packaged_protocol["implementation_files"] != implementation_files:
        raise RuntimeError("installed protocol source manifest mismatch")

    summary = {
        "compute_backend": jax.default_backend(),
        "device_count": len(jax.devices()),
        "device_kinds": [device.device_kind for device in jax.devices()],
        "implementation_digest": implementation_digest,
        "jax_enable_x64": True,
        "package_versions": observed_packages,
        "packaged_protocol_path": str(packaged_protocol_path),
        "protocol_artifact_digest": packaged_protocol["artifact_digest"],
        "python_major_minor": "3.12",
        "study_fingerprint": spec.fingerprint,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


RUNTIME_VALIDATION = (
    _validate_runtime_environment() if ACTION in {"preflight", "run"} else None
)
if ACTION == "preflight":
    print("Preflight complete. No CFD execution was performed.")

# COMMAND ----------


def _write_immutable_json(path: Path, body: dict[str, object]) -> None:
    payload = {**body, "artifact_digest": _digest(body)}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical notebook artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


RUN_SUMMARY: dict[str, object] | None = None
if ACTION == "run":
    if CONFIRMATION != CONFIRMATION_TOKEN:
        raise RuntimeError("confirmation token mismatch; primary Databricks replication was not started")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"the reviewed environment requires Python 3.12, observed {sys.version.split()[0]}")

    import platform
    import shutil

    from codex_hydrogym.gate0 import ensemble_replication

    if RUNTIME_VALIDATION is None:
        raise RuntimeError("runtime validation did not run")
    observed_packages = RUNTIME_VALIDATION["package_versions"]
    packaged_protocol_path = Path(str(RUNTIME_VALIDATION["packaged_protocol_path"]))
    packaged_protocol = json.loads(packaged_protocol_path.read_text(encoding="utf-8"))

    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_absolute() or OUTPUT_DIR.startswith("dbfs:"):
        raise RuntimeError("output_dir must be an absolute /Workspace, /Volumes, or local path")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_protocol = output_dir / "protocol.json"
    if output_protocol.exists():
        if output_protocol.read_bytes() != packaged_protocol_path.read_bytes():
            raise RuntimeError("output directory contains a different protocol")
    else:
        temporary_protocol = output_dir / f".protocol.json.{os.getpid()}.tmp"
        shutil.copyfile(packaged_protocol_path, temporary_protocol)
        os.replace(temporary_protocol, output_protocol)

    execution_context = {
        "claim_role": "primary_decision_bearing_execution",
        "accelerator_count": 1,
        "accelerator_type": "GPU_1xH100",
        "compute_backend": "gpu",
        "decision_bearing_execution": True,
        "execution_backend_amendment_artifact_digest": BACKEND_AMENDMENT_ARTIFACT_DIGEST,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "jax_enable_x64": True,
        "local_partial_artifact_in_analysis": False,
        "notebook_schema": "codex_hydrogym.ensemble_replication_databricks.v2",
        "package_versions": observed_packages,
        "platform": platform.platform(),
        "platform_transition_artifact_digest": TRANSITION_ARTIFACT_DIGEST,
        "prior_or_local_results_in_analysis": 0,
        "python_major_minor": "3.12",
        "sole_analysis_set": True,
        "study_fingerprint": STUDY_FINGERPRINT,
        "wheel_sha256": WHEEL_SHA256,
    }
    _write_immutable_json(output_dir / "databricks_execution_context.json", execution_context)

    exit_code = ensemble_replication.main(["--stage", "run", "--output-dir", str(output_dir)])
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    RUN_SUMMARY = {
        "artifact_digest": result["artifact_digest"],
        "claim_role": "primary_decision_bearing_execution",
        "exit_code": exit_code,
        "output_dir": str(output_dir),
        "study_fingerprint": result["study_fingerprint"],
        "supports_designing_full_gate": result["analysis"]["supports_designing_full_gate"],
    }
    print(json.dumps(RUN_SUMMARY, indent=2, sort_keys=True))
elif ACTION == "review":
    print("Run step skipped because action=review.")

# COMMAND ----------


if RUN_SUMMARY is not None:
    dbutils.notebook.exit(json.dumps(RUN_SUMMARY, sort_keys=True))
if ACTION == "preflight" and RUNTIME_VALIDATION is not None:
    dbutils.notebook.exit(
        json.dumps(
            {
                "action": "preflight",
                "backend_amendment": BACKEND_AMENDMENT_REVIEW,
                "cfds_executed": 0,
                "runtime": RUNTIME_VALIDATION,
                "transition": TRANSITION_REVIEW,
                "wheel": WHEEL_REVIEW,
            },
            sort_keys=True,
        )
    )
if ACTION == "review":
    dbutils.notebook.exit(
        json.dumps(
            {
                "action": "review",
                "backend_amendment": BACKEND_AMENDMENT_REVIEW,
                "cfds_executed": 0,
                "transition": TRANSITION_REVIEW,
                "wheel": WHEEL_REVIEW,
            },
            sort_keys=True,
        )
    )

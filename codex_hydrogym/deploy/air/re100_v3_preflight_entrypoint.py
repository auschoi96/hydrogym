"""Zero-CFD one-H100 AI Runtime preflight for the frozen Re=100 Gate 0 v3."""

from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform


STUDY_FINGERPRINT = "885ff77559dadd18cc54d91a30ecb6a48477a4c2baed46fc728635ea3eae8b38"
IMPLEMENTATION_DIGEST = "17fd18a51e8bfb2e8b6d018e7fe824a9b68921fe38d62d231e6634d6203b9dfe"
PROTOCOL_ARTIFACT_DIGEST = "024039795a851caa0a1ea77580983aa2c869d40d05564f0765fdc56f1920db3f"
PROTOCOL_RAW_SHA256 = "2dc2ee5ac7287d74e99d45bcd69ce7c3af9ef5403d7d3c32ea81e0ec816a74d6"
WHEEL_SHA256 = "e381b42d415b0644fd773be67ef9aab94133289e6559933bf72e079da50e2e51"
PROTOCOL_RELATIVE_PATH = "evidence/gate0_v3/885ff77559da-17fd18a51e8b/protocol.json"
PINNED_PACKAGES = {
    "chex": "0.1.92",
    "flax": "0.12.0",
    "gymnasium": "1.3.0",
    "gymnax": "0.0.9",
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


def _load_protocol(path: Path) -> dict[str, object]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != PROTOCOL_RAW_SHA256:
        raise RuntimeError("v3 protocol raw SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if payload.get("artifact_digest") != _digest(body):
        raise RuntimeError("v3 protocol canonical digest mismatch")
    if payload.get("artifact_digest") != PROTOCOL_ARTIFACT_DIGEST:
        raise RuntimeError("v3 protocol artifact identity mismatch")
    return payload


def _attach_mlflow() -> tuple[object, bool]:
    import mlflow

    owns_run = mlflow.active_run() is None
    if owns_run:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name="re100_gate0_v3_zero_cfd_h100_preflight")
    return mlflow, owns_run


def _end_owned_run(mlflow: object, *, owns_run: bool, status: str) -> None:
    if owns_run and getattr(mlflow, "active_run")() is not None:
        getattr(mlflow, "end_run")(status=status)


def _validate() -> tuple[dict[str, object], Path]:
    if os.sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"v3 AIR preflight requires Python 3.12, observed {platform.python_version()}")
    source_root = Path(os.environ["CODE_SOURCE_PATH"])
    protocol_path = source_root / PROTOCOL_RELATIVE_PATH
    protocol = _load_protocol(protocol_path)
    wheel_path = Path(os.environ["CODEX_HYDROGYM_V3_WHEEL_PATH"])
    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if wheel_digest != WHEEL_SHA256:
        raise RuntimeError("v3 wheel SHA-256 mismatch")

    observed_packages = {name: package_version(name) for name in PINNED_PACKAGES}
    if observed_packages != PINNED_PACKAGES:
        raise RuntimeError("v3 AIR package versions differ from the frozen pins")

    import jax

    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if not jax.config.x64_enabled:
        raise RuntimeError("v3 AIR preflight could not enable JAX float64")
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("v3 AIR preflight requires GPU-backed JAX")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError(f"v3 AIR preflight requires exactly one H100, observed {devices}")

    from codex_hydrogym.gate0 import re100_v3

    spec = re100_v3.Re100Gate0V3Spec()
    implementation_files, implementation_digest = re100_v3._implementation_manifest_v3()
    if spec.fingerprint != STUDY_FINGERPRINT:
        raise RuntimeError("installed v3 study fingerprint mismatch")
    if implementation_digest != IMPLEMENTATION_DIGEST:
        raise RuntimeError("installed v3 implementation digest mismatch")
    if protocol.get("study_fingerprint") != spec.fingerprint:
        raise RuntimeError("v3 protocol study fingerprint mismatch")
    if protocol.get("implementation_digest") != implementation_digest:
        raise RuntimeError("v3 protocol implementation digest mismatch")
    if protocol.get("implementation_files") != implementation_files:
        raise RuntimeError("v3 protocol source manifest mismatch")
    if protocol.get("execution_authorized") is not False:
        raise RuntimeError("v3 preflight requires execution to remain unauthorized")
    if protocol.get("reserved_cases_opened") is not False:
        raise RuntimeError("v3 preflight requires reserved cases to remain unopened")
    if protocol.get("rl_training_performed") is not False:
        raise RuntimeError("v3 preflight protocol cannot contain RL training")
    if any((protocol_path.parent / name).exists() for name in (
        "condition_base.json",
        "condition_temporal.json",
        "condition_spatial.json",
        "result.json",
    )):
        raise RuntimeError("v3 preflight source namespace contains execution artifacts")
    if re100_v3.main(
        ["--stage", "review", "--output-dir", str(protocol_path.parent)]
    ) != 0:
        raise RuntimeError("installed v3 protocol review failed")

    summary = {
        "action": "re100_gate0_v3_h100_preflight",
        "accelerator_count": 1,
        "accelerator_type": "GPU_1xH100",
        "cfds_executed": 0,
        "compute_backend": jax.default_backend(),
        "device_kinds": [device.device_kind for device in devices],
        "execution_authorized": False,
        "expected_trajectory_count": spec.expected_trajectory_count,
        "implementation_digest": implementation_digest,
        "jax_enable_x64": True,
        "package_versions": observed_packages,
        "prior_or_local_observations_in_analysis": 0,
        "protocol_artifact_digest": protocol["artifact_digest"],
        "protocol_raw_sha256": PROTOCOL_RAW_SHA256,
        "reserved_cases_opened": False,
        "rl_training_performed": False,
        "study_fingerprint": spec.fingerprint,
        "wheel_sha256": wheel_digest,
    }
    return summary, protocol_path


def main() -> int:
    summary, protocol_path = _validate()
    mlflow, owns_run = _attach_mlflow()
    try:
        mlflow.set_tags(
            {
                "codex_hydrogym.claim_role": "gate0_v3_preflight_only",
                "codex_hydrogym.execution_backend": "GPU_1xH100",
                "codex_hydrogym.study_fingerprint": STUDY_FINGERPRINT,
                "codex_hydrogym.workflow": "re100_gate0_v3_preflight",
            }
        )
        mlflow.log_dict(summary, "gate0_v3/preflight.json")
        mlflow.log_artifact(str(protocol_path), artifact_path="gate0_v3/frozen_protocol")
        print("CODEX_HYDROGYM_V3_PREFLIGHT_JSON=" + _canonical(summary), flush=True)
    except BaseException:
        try:
            _end_owned_run(mlflow, owns_run=owns_run, status="FAILED")
        except Exception as teardown_error:
            print(
                "CODEX_HYDROGYM_V3_MLFLOW_TEARDOWN_ERROR=" + repr(teardown_error),
                file=os.sys.stderr,
                flush=True,
            )
        raise
    else:
        _end_owned_run(mlflow, owns_run=owns_run, status="FINISHED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

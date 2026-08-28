"""Fail-closed one-H100 AIR entry point for the held-out Re=100 Gate 0 v3."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Mapping


STUDY_FINGERPRINT = "885ff77559dadd18cc54d91a30ecb6a48477a4c2baed46fc728635ea3eae8b38"
IMPLEMENTATION_DIGEST = "17fd18a51e8bfb2e8b6d018e7fe824a9b68921fe38d62d231e6634d6203b9dfe"
PROTOCOL_ARTIFACT_DIGEST = "024039795a851caa0a1ea77580983aa2c869d40d05564f0765fdc56f1920db3f"
PROTOCOL_RAW_SHA256 = "2dc2ee5ac7287d74e99d45bcd69ce7c3af9ef5403d7d3c32ea81e0ec816a74d6"
REVIEW_ATTESTATION_ARTIFACT_DIGEST = (
    "39b4ab964755ff1ea1d7747939ac77517219faa2648702adbc4d022040902667"
)
REVIEW_ATTESTATION_RAW_SHA256 = (
    "4dba8cab5f561ce3800915d3aaa0d5827f33afba526282d1f0b48f758c7a605d"
)
EXECUTION_TOKEN_SHA256 = "f6cf4be64101d4642489de6bd9c9558c67dfb152c3795c56471a3d0a237b51c5"
WHEEL_SHA256 = "e381b42d415b0644fd773be67ef9aab94133289e6559933bf72e079da50e2e51"
PROTOCOL_RELATIVE_PATH = "evidence/gate0_v3/885ff77559da-17fd18a51e8b/protocol.json"
REVIEW_ATTESTATION_RELATIVE_PATH = (
    "evidence/gate0_v3/885ff77559da-17fd18a51e8b/review_attestation.json"
)
OUTPUT_NAMESPACE = (
    "/Workspace/Users/austin.choi@databricks.com/codex_hydrogym_gate0_v3/"
    "evidence/885ff77559da-17fd18a51e8b"
)
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


def _load_digest_bound_json(
    path: Path,
    *,
    raw_sha256: str,
    artifact_digest: str,
) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != raw_sha256:
        raise RuntimeError(f"raw SHA-256 mismatch for {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if payload.get("artifact_digest") != _digest(body):
        raise RuntimeError(f"canonical artifact digest mismatch for {path.name}")
    if payload.get("artifact_digest") != artifact_digest:
        raise RuntimeError(f"artifact identity mismatch for {path.name}")
    return payload


def _write_immutable_json(path: Path, body: Mapping[str, object]) -> dict[str, object]:
    if "artifact_digest" in body:
        raise ValueError("immutable artifact bodies must not define artifact_digest")
    payload = {**body, "artifact_digest": _digest(body)}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical v3 artifact: {path}")
        return payload
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    return payload


def _copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"refusing to overwrite non-identical v3 artifact: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _attach_mlflow() -> tuple[object, bool]:
    import mlflow

    owns_run = mlflow.active_run() is None
    if owns_run:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name="re100_gate0_v3_one_full_execution")
    return mlflow, owns_run


def _end_owned_run(mlflow: object, *, owns_run: bool, status: str) -> None:
    if owns_run and getattr(mlflow, "active_run")() is not None:
        getattr(mlflow, "end_run")(status=status)


def _validate() -> tuple[dict[str, object], object, Path, Path, Path]:
    action = os.environ.get("CODEX_HYDROGYM_V3_ACTION", "authorization_preflight")
    if action not in {"authorization_preflight", "run"}:
        raise RuntimeError(f"unsupported v3 AIR action: {action}")
    if os.sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"v3 AIR execution requires Python 3.12, observed {platform.python_version()}")

    source_root = Path(os.environ["CODE_SOURCE_PATH"])
    protocol_path = source_root / PROTOCOL_RELATIVE_PATH
    review_path = source_root / REVIEW_ATTESTATION_RELATIVE_PATH
    protocol = _load_digest_bound_json(
        protocol_path,
        raw_sha256=PROTOCOL_RAW_SHA256,
        artifact_digest=PROTOCOL_ARTIFACT_DIGEST,
    )
    review = _load_digest_bound_json(
        review_path,
        raw_sha256=REVIEW_ATTESTATION_RAW_SHA256,
        artifact_digest=REVIEW_ATTESTATION_ARTIFACT_DIGEST,
    )

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
        raise RuntimeError("v3 AIR execution could not enable JAX float64")
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("v3 AIR execution requires GPU-backed JAX")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError(f"v3 AIR execution requires exactly one H100, observed {devices}")

    from codex_hydrogym.gate0 import re100_v3

    spec = re100_v3.Re100Gate0V3Spec()
    implementation_files, implementation_digest = re100_v3._implementation_manifest_v3()
    if spec.fingerprint != STUDY_FINGERPRINT:
        raise RuntimeError("installed v3 study fingerprint mismatch")
    if implementation_digest != IMPLEMENTATION_DIGEST:
        raise RuntimeError("installed v3 implementation digest mismatch")
    if protocol.get("study_fingerprint") != spec.fingerprint:
        raise RuntimeError("v3 protocol study mismatch")
    if protocol.get("implementation_digest") != implementation_digest:
        raise RuntimeError("v3 protocol implementation mismatch")
    if protocol.get("implementation_files") != implementation_files:
        raise RuntimeError("v3 protocol source manifest mismatch")
    if protocol.get("execution_authorized") is not False:
        raise RuntimeError("v3 authorization must remain external to the frozen protocol")
    if protocol.get("reserved_cases_opened") is not False:
        raise RuntimeError("v3 protocol says reserved cases were already opened")
    if protocol.get("rl_training_performed") is not False:
        raise RuntimeError("v3 protocol cannot contain RL training")

    validated_review = re100_v3._validate_review_attestation(
        review_path,
        protocol=protocol,
        spec=spec,
        implementation_digest=implementation_digest,
    )
    if validated_review.get("artifact_digest") != REVIEW_ATTESTATION_ARTIFACT_DIGEST:
        raise RuntimeError("installed v3 review attestation identity mismatch")
    token = os.environ.get("CODEX_HYDROGYM_GATE0_V3_EXECUTION_TOKEN")
    token_digest = hashlib.sha256((token or "").encode("utf-8")).hexdigest()
    if not token or not hmac.compare_digest(token_digest, EXECUTION_TOKEN_SHA256):
        raise RuntimeError("v3 execution token does not match the approved attestation")
    if validated_review.get("execution_token_sha256") != token_digest:
        raise RuntimeError("v3 review attestation and external token differ")

    output_dir = Path(os.environ["CODEX_HYDROGYM_V3_OUTPUT_DIR"])
    if str(output_dir) != OUTPUT_NAMESPACE or not output_dir.is_absolute():
        raise RuntimeError("v3 AIR output namespace differs from the reviewed namespace")
    if not output_dir.is_dir():
        raise RuntimeError("v3 AIR output namespace does not exist")
    observed_names = sorted(path.name for path in output_dir.iterdir())
    if observed_names != ["protocol.json"]:
        raise RuntimeError(f"v3 AIR pre-execution namespace is not empty: {observed_names}")
    output_protocol_path = output_dir / "protocol.json"
    if output_protocol_path.read_bytes() != protocol_path.read_bytes():
        raise RuntimeError("v3 workspace protocol differs from the frozen source protocol")
    if re100_v3.main(["--stage", "review", "--output-dir", str(output_dir)]) != 0:
        raise RuntimeError("installed v3 protocol review failed")

    summary = {
        "action": action,
        "accelerator_count": 1,
        "accelerator_type": "GPU_1xH100",
        "cfds_executed_before_runner": 0,
        "compute_backend": jax.default_backend(),
        "device_kinds": [device.device_kind for device in devices],
        "execution_authorized_by_external_attestation": True,
        "execution_token_sha256": token_digest,
        "expected_trajectory_count": spec.expected_trajectory_count,
        "expected_window_count": spec.expected_window_count,
        "implementation_digest": implementation_digest,
        "jax_enable_x64": True,
        "output_namespace": str(output_dir),
        "package_versions": observed_packages,
        "pre_execution_namespace_entries": observed_names,
        "prior_or_local_observations_in_analysis": 0,
        "protocol_artifact_digest": protocol["artifact_digest"],
        "protocol_raw_sha256": PROTOCOL_RAW_SHA256,
        "reserved_cases_opened_before_runner": False,
        "review_attestation_artifact_digest": validated_review["artifact_digest"],
        "review_attestation_raw_sha256": REVIEW_ATTESTATION_RAW_SHA256,
        "rl_training_performed": False,
        "study_fingerprint": spec.fingerprint,
        "wheel_sha256": wheel_digest,
    }
    return summary, re100_v3, protocol_path, review_path, output_dir


def main() -> int:
    validation, re100_v3, protocol_path, review_path, output_dir = _validate()
    action = str(validation["action"])
    mlflow, owns_run = _attach_mlflow()
    try:
        claim_role = (
            "gate0_v3_authorization_preflight_only"
            if action == "authorization_preflight"
            else "gate0_v3_primary_decision_bearing_execution"
        )
        mlflow.set_tags(
            {
                "codex_hydrogym.claim_role": claim_role,
                "codex_hydrogym.execution_backend": "GPU_1xH100",
                "codex_hydrogym.study_fingerprint": STUDY_FINGERPRINT,
                "codex_hydrogym.workflow": "re100_gate0_v3",
            }
        )
        mlflow.log_dict(validation, "gate0_v3/authorization_preflight.json")
        mlflow.log_artifact(str(protocol_path), artifact_path="gate0_v3/frozen_protocol")
        mlflow.log_artifact(str(review_path), artifact_path="gate0_v3/review")
        print("CODEX_HYDROGYM_V3_AUTHORIZATION_JSON=" + _canonical(validation), flush=True)
        if action == "run":
            output_review_path = output_dir / "review_attestation.json"
            _copy_immutable(review_path, output_review_path)
            execution_context = _write_immutable_json(
                output_dir / "databricks_execution_context.json",
                {
                    "accelerator_count": 1,
                    "accelerator_type": "GPU_1xH100",
                    "claim_role": claim_role,
                    "compute_backend": "gpu",
                    "decision_bearing_execution": True,
                    "execution_started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "execution_token_sha256": EXECUTION_TOKEN_SHA256,
                    "implementation_digest": IMPLEMENTATION_DIGEST,
                    "jax_enable_x64": True,
                    "job_id": os.environ.get("DATABRICKS_JOB_ID"),
                    "job_run_id": os.environ.get("DATABRICKS_RUN_ID"),
                    "package_versions": validation["package_versions"],
                    "precision": "float64",
                    "prior_or_local_observations_in_analysis": 0,
                    "protocol_artifact_digest": PROTOCOL_ARTIFACT_DIGEST,
                    "review_attestation_artifact_digest": (
                        REVIEW_ATTESTATION_ARTIFACT_DIGEST
                    ),
                    "runner_schema_version": "codex_hydrogym.gate0.re100_v3.air.v1",
                    "sole_analysis_set": True,
                    "study_fingerprint": STUDY_FINGERPRINT,
                    "wheel_sha256": WHEEL_SHA256,
                },
            )
            runner_exit_code = re100_v3.main(
                [
                    "--stage",
                    "run",
                    "--output-dir",
                    str(output_dir),
                    "--review-attestation",
                    str(output_review_path),
                ]
            )
            result_path = output_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            re100_v3._validate_artifact(result)
            analysis = result.get("analysis")
            if not isinstance(analysis, dict) or type(analysis.get("passed")) is not bool:
                raise RuntimeError("v3 result is missing its frozen decision")
            run_summary = _write_immutable_json(
                output_dir / "air_run_summary.json",
                {
                    "schema_version": "codex_hydrogym.gate0.re100_v3.air_summary.v1",
                    "claim_role": claim_role,
                    "condition_artifact_digests": result["condition_artifact_digests"],
                    "gate_passed": analysis["passed"],
                    "heldout_gate_performed": analysis["heldout_gate_performed"],
                    "implementation_digest": IMPLEMENTATION_DIGEST,
                    "output_namespace": str(output_dir),
                    "prior_or_local_observations_in_analysis": 0,
                    "protocol_artifact_digest": PROTOCOL_ARTIFACT_DIGEST,
                    "reserved_cases_opened": True,
                    "result_artifact_digest": result["artifact_digest"],
                    "review_attestation_artifact_digest": (
                        REVIEW_ATTESTATION_ARTIFACT_DIGEST
                    ),
                    "rl_training_performed": False,
                    "runner_exit_code": runner_exit_code,
                    "study_fingerprint": STUDY_FINGERPRINT,
                    "trajectory_count": result["trajectory_count"],
                    "window_count": result["window_count"],
                },
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="gate0_v3/full_execution")
            mlflow.log_metric("gate0_v3.passed", float(bool(analysis["passed"])))
            mlflow.log_metric("gate0_v3.trajectory_count", float(result["trajectory_count"]))
            mlflow.log_metric("gate0_v3.window_count", float(result["window_count"]))
            print("CODEX_HYDROGYM_V3_RESULT_JSON=" + _canonical(run_summary), flush=True)
            print(
                "CODEX_HYDROGYM_V3_EXECUTION_CONTEXT_JSON="
                + _canonical(execution_context),
                flush=True,
            )
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

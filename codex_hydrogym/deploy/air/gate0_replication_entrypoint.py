"""Fail-closed Databricks AIR entry point for the frozen Gate 0 replication."""

from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import shutil
import zipfile


STUDY_FINGERPRINT = "269507101a5206fccab3c90504f7a46009f28381070a0d97875a06429fb19b62"
IMPLEMENTATION_DIGEST = "a5ab894e5ff4d3b669da274771f247e58f06aab992873fc9fe76dfdcf8622d8c"
PROTOCOL_ARTIFACT_DIGEST = "3914aedc99979693bf693772a56eef83c3c242c6cd72dc7fda8c07583d781c87"
WHEEL_SHA256 = "91ae939efbacfbd8e3e3aedcf07d1c1e02f9dac642e7d8d381c107ba6505ddc1"
TRANSITION_SHA256 = "69f000608ae8b0aa9b0d8f3433ce1108617936e20f9763d77fe398ad5c96a428"
TRANSITION_ARTIFACT_DIGEST = "deb256b550dd7d3d0fc88746db4ca7b0cbcdeb60f8b921d4fe19bb0466ad2e8a"
BACKEND_AMENDMENT_SHA256 = "ec39d51fdbe288080a52730f5d665acea4ee8bcb1ee00c26f7900a2107156bdf"
BACKEND_AMENDMENT_ARTIFACT_DIGEST = (
    "28747c56c53d2dd251dee8f17f49ef1c67c5d5b01d5fc9b3ea9d1b8f8c84e181"
)
PACKAGED_PROTOCOL = "codex_hydrogym/evidence/ensemble_replication/269507101a52-a5ab894e5ff4/protocol.json"
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


def _load_digest_bound_json(path: Path, *, raw_sha256: str, artifact_digest: str) -> dict[str, object]:
    observed_raw_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_raw_digest != raw_sha256:
        raise RuntimeError(f"raw SHA-256 mismatch for {path.name}: {observed_raw_digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed_artifact_digest = payload.get("artifact_digest")
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    if observed_artifact_digest != _digest(body) or observed_artifact_digest != artifact_digest:
        raise RuntimeError(f"artifact digest mismatch for {path.name}")
    return payload


def _write_immutable_json(path: Path, body: dict[str, object]) -> dict[str, object]:
    if "artifact_digest" in body:
        raise ValueError("immutable artifact bodies must not define artifact_digest")
    payload = {**body, "artifact_digest": _digest(body)}
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite non-identical AIR artifact: {path}")
        return payload
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    return payload


def _attach_mlflow() -> tuple[object, bool]:
    import mlflow

    owns_run = mlflow.active_run() is None
    if owns_run:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name="re100_ten_seed_replication_air")
    return mlflow, owns_run


def _end_owned_mlflow_run(mlflow: object, *, owns_run: bool, status: str) -> None:
    if not owns_run:
        return
    active_run = getattr(mlflow, "active_run")()
    if active_run is not None:
        getattr(mlflow, "end_run")(status=status)


def _air_run_summary_body(
    result: dict[str, object],
    *,
    output_dir: Path,
    runner_exit_code: int,
) -> dict[str, object]:
    analysis = result.get("analysis")
    if not isinstance(analysis, dict):
        raise RuntimeError("replication result is missing its analysis")
    result_artifact_digest = result.get("artifact_digest")
    if not isinstance(result_artifact_digest, str) or len(result_artifact_digest) != 64:
        raise RuntimeError("replication result artifact digest is invalid")
    return {
        "schema_version": "codex_hydrogym.ensemble_replication_air_summary.v2",
        "result_artifact_digest": result_artifact_digest,
        "claim_role": "primary_decision_bearing_execution",
        "output_dir": str(output_dir),
        "runner_exit_code": runner_exit_code,
        "study_fingerprint": result["study_fingerprint"],
        "supports_designing_full_gate": analysis["supports_designing_full_gate"],
    }


def _validate() -> tuple[dict[str, object], object, Path, dict[str, object]]:
    if os.sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"AIR execution requires Python 3.12, observed {platform.python_version()}")

    source_root = Path(os.environ["CODE_SOURCE_PATH"])
    wheel_path = source_root / "notebooks" / "artifacts" / "hydrogym-1.0.0-py3-none-any.whl"
    evidence_root = (
        source_root
        / "evidence"
        / "ensemble_replication"
        / "269507101a52-a5ab894e5ff4"
    )
    transition_path = evidence_root / "platform_transition.json"
    backend_amendment_path = evidence_root / "execution_backend_amendment.json"

    wheel_digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    if wheel_digest != WHEEL_SHA256:
        raise RuntimeError(f"wheel SHA-256 mismatch: {wheel_digest}")
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_protocol = json.loads(archive.read(PACKAGED_PROTOCOL))
        wheel_protocol_body = {
            key: value for key, value in wheel_protocol.items() if key != "artifact_digest"
        }
        if (
            wheel_protocol.get("artifact_digest") != _digest(wheel_protocol_body)
            or wheel_protocol.get("artifact_digest") != PROTOCOL_ARTIFACT_DIGEST
        ):
            raise RuntimeError("wheel protocol artifact digest mismatch")
        if wheel_protocol.get("study_fingerprint") != STUDY_FINGERPRINT:
            raise RuntimeError("wheel protocol study fingerprint mismatch")
        if wheel_protocol.get("implementation_digest") != IMPLEMENTATION_DIGEST:
            raise RuntimeError("wheel protocol implementation digest mismatch")
        for relative_path, expected_digest in wheel_protocol["implementation_files"].items():
            observed_digest = hashlib.sha256(archive.read(relative_path)).hexdigest()
            if observed_digest != expected_digest:
                raise RuntimeError(f"wheel source hash mismatch: {relative_path}")

    transition = _load_digest_bound_json(
        transition_path,
        raw_sha256=TRANSITION_SHA256,
        artifact_digest=TRANSITION_ARTIFACT_DIGEST,
    )
    backend_amendment = _load_digest_bound_json(
        backend_amendment_path,
        raw_sha256=BACKEND_AMENDMENT_SHA256,
        artifact_digest=BACKEND_AMENDMENT_ARTIFACT_DIGEST,
    )
    if transition.get("study_fingerprint") != STUDY_FINGERPRINT:
        raise RuntimeError("platform transition study fingerprint mismatch")
    if transition["local_execution"]["partial_artifact"]["content_or_metrics_inspected"] is not False:
        raise RuntimeError("local partial artifact is not recorded as blind")
    databricks_execution = transition["databricks_execution"]
    output_dir = os.environ["CODEX_HYDROGYM_OUTPUT_DIR"]
    if databricks_execution["output_namespace"] != output_dir:
        raise RuntimeError("AIR output path differs from the frozen Databricks output namespace")
    if databricks_execution["prior_or_local_results_in_analysis"] != 0:
        raise RuntimeError("platform transition does not exclude prior/local results")
    if databricks_execution["sole_analysis_set"] is not True:
        raise RuntimeError("Databricks execution is not recorded as the sole analysis set")

    if backend_amendment.get("previous_transition_artifact_digest") != TRANSITION_ARTIFACT_DIGEST:
        raise RuntimeError("backend amendment does not bind the platform transition")
    expected_backend = {
        "accelerator_count": 1,
        "accelerator_type": "GPU_1xH100",
        "execution_service": "Databricks AI Runtime",
        "jax_backend": "gpu",
        "precision": "float64",
    }
    if backend_amendment.get("authorized_change") != expected_backend:
        raise RuntimeError("backend amendment differs from the reviewed H100 contract")

    observed_packages = {name: package_version(name) for name in PINNED_PACKAGES}
    if observed_packages != PINNED_PACKAGES:
        raise RuntimeError("installed packages differ from pins: " + json.dumps(observed_packages, sort_keys=True))

    import jax

    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX float64 could not be enabled")
    devices = jax.devices()
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in devices):
        raise RuntimeError("AIR execution requires GPU-backed JAX")
    if len(devices) != 1 or "H100" not in devices[0].device_kind.upper():
        raise RuntimeError(f"AIR execution requires exactly one H100, observed: {devices}")

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
        "action": os.environ.get("CODEX_HYDROGYM_ACTION", "preflight"),
        "backend_amendment_artifact_digest": BACKEND_AMENDMENT_ARTIFACT_DIGEST,
        "compute_backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kinds": [device.device_kind for device in devices],
        "implementation_digest": implementation_digest,
        "jax_enable_x64": True,
        "package_versions": observed_packages,
        "platform_transition_artifact_digest": TRANSITION_ARTIFACT_DIGEST,
        "protocol_artifact_digest": packaged_protocol["artifact_digest"],
        "python_major_minor": "3.12",
        "study_fingerprint": spec.fingerprint,
        "wheel_sha256": wheel_digest,
    }
    return summary, ensemble_replication, packaged_protocol_path, packaged_protocol


def main() -> int:
    action = os.environ.get("CODEX_HYDROGYM_ACTION", "preflight")
    if action not in {"preflight", "run"}:
        raise RuntimeError(f"unsupported CODEX_HYDROGYM_ACTION: {action}")

    validation, ensemble_replication, packaged_protocol_path, packaged_protocol = _validate()
    mlflow, owns_mlflow_run = _attach_mlflow()
    try:
        mlflow.set_tags(
            {
                "codex_hydrogym.claim_role": "primary_decision_bearing_execution",
                "codex_hydrogym.execution_backend": "GPU_1xH100",
                "codex_hydrogym.study_fingerprint": STUDY_FINGERPRINT,
                "codex_hydrogym.workflow": "ensemble_replication",
            }
        )
        mlflow.log_dict(validation, "gate0/preflight.json")
        print("CODEX_HYDROGYM_PREFLIGHT_JSON=" + _canonical(validation), flush=True)
        if action == "run":
            output_dir = Path(os.environ["CODEX_HYDROGYM_OUTPUT_DIR"])
            if not output_dir.is_absolute() or str(output_dir).startswith("dbfs:"):
                raise RuntimeError("AIR output path must be an absolute /Workspace or /Volumes path")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_protocol = output_dir / "protocol.json"
            if output_protocol.exists():
                if output_protocol.read_bytes() != packaged_protocol_path.read_bytes():
                    raise RuntimeError("AIR output directory contains a different protocol")
            else:
                temporary_protocol = output_dir / f".protocol.json.{os.getpid()}.tmp"
                shutil.copyfile(packaged_protocol_path, temporary_protocol)
                os.replace(temporary_protocol, output_protocol)

            execution_context = {
                "accelerator_count": 1,
                "accelerator_type": "GPU_1xH100",
                "claim_role": "primary_decision_bearing_execution",
                "compute_backend": "gpu",
                "decision_bearing_execution": True,
                "execution_backend_amendment_artifact_digest": BACKEND_AMENDMENT_ARTIFACT_DIGEST,
                "implementation_digest": IMPLEMENTATION_DIGEST,
                "jax_enable_x64": True,
                "local_partial_artifact_in_analysis": False,
                "package_versions": validation["package_versions"],
                "platform": platform.platform(),
                "platform_transition_artifact_digest": TRANSITION_ARTIFACT_DIGEST,
                "prior_or_local_results_in_analysis": 0,
                "python_major_minor": "3.12",
                "runner_schema": "codex_hydrogym.ensemble_replication_air.v1",
                "sole_analysis_set": True,
                "study_fingerprint": STUDY_FINGERPRINT,
                "wheel_sha256": WHEEL_SHA256,
            }
            _write_immutable_json(output_dir / "databricks_execution_context.json", execution_context)

            runner_exit_code = ensemble_replication.main(
                ["--stage", "run", "--output-dir", str(output_dir)]
            )
            result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            run_summary = _write_immutable_json(
                output_dir / "air_run_summary.json",
                _air_run_summary_body(
                    result,
                    output_dir=output_dir,
                    runner_exit_code=runner_exit_code,
                ),
            )
            mlflow.log_artifacts(str(output_dir), artifact_path="gate0/ensemble_replication")
            mlflow.log_metric(
                "gate0.supports_designing_full_gate",
                float(bool(run_summary["supports_designing_full_gate"])),
            )
            print("CODEX_HYDROGYM_RESULT_JSON=" + _canonical(run_summary), flush=True)
    except BaseException:
        try:
            _end_owned_mlflow_run(mlflow, owns_run=owns_mlflow_run, status="FAILED")
        except Exception as teardown_error:
            print(
                "CODEX_HYDROGYM_MLFLOW_TEARDOWN_ERROR=" + repr(teardown_error),
                file=os.sys.stderr,
                flush=True,
            )
        raise
    else:
        _end_owned_mlflow_run(mlflow, owns_run=owns_mlflow_run, status="FINISHED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

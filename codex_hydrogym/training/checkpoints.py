"""Integrity-checked, non-pickle checkpoints for codex_hydrogym PPO runs."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from flax import serialization
import jax
import numpy as np

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig, config_fingerprint
from codex_hydrogym.training.ppo import RunnerState, make_initialize


CHECKPOINT_FORMAT = "codex_hydrogym.flax_runner_state.v1"
MANIFEST_NAME = "manifest.json"
_STATE_NAME = re.compile(r"state-[0-9a-f]{16}\.msgpack")


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint bytes or manifest are corrupt or unsafe."""


class CheckpointCompatibilityError(CheckpointError):
    """The checkpoint does not match the requested training configuration."""


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    checkpoint_directory: str | Path,
    runner_state: RunnerState,
    config: KolmogorovPPOConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish a checksummed Flax MessagePack checkpoint.

    Publishing the manifest last makes an interrupted write invisible to readers.
    Existing checkpoints are preserved unless ``overwrite=True`` is explicit.
    """
    directory = Path(checkpoint_directory)
    manifest_path = directory / MANIFEST_NAME
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"checkpoint path is not a directory: {directory}")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {directory}")
    if directory.exists() and any(directory.iterdir()) and not manifest_path.exists():
        raise FileExistsError(f"refusing to write into a non-checkpoint directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    host_state = jax.device_get(runner_state)
    payload = serialization.to_bytes(host_state)
    checksum = hashlib.sha256(payload).hexdigest()
    state_name = f"state-{checksum[:16]}.msgpack"
    _atomic_write(directory / state_name, payload)

    manifest = {
        "format": CHECKPOINT_FORMAT,
        "project_label": PROJECT_LABEL,
        "config_fingerprint": config_fingerprint(config),
        "completed_updates": int(np.asarray(host_state.completed_updates)),
        "state_file": state_name,
        "state_sha256": checksum,
        "state_bytes": len(payload),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(manifest_path, manifest_bytes)
    return directory


def _read_manifest(directory: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointIntegrityError(f"cannot read checkpoint manifest in {directory}") from error
    if not isinstance(manifest, dict):
        raise CheckpointIntegrityError("checkpoint manifest must be a JSON object")
    return manifest


def restore_checkpoint(
    checkpoint_directory: str | Path,
    config: KolmogorovPPOConfig,
    *,
    target_state: RunnerState | None = None,
) -> RunnerState:
    """Verify and restore a checkpoint into a compatible RunnerState template."""
    directory = Path(checkpoint_directory)
    manifest = _read_manifest(directory)

    if manifest.get("format") != CHECKPOINT_FORMAT or manifest.get("project_label") != PROJECT_LABEL:
        raise CheckpointCompatibilityError("checkpoint format or project label is incompatible")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise CheckpointCompatibilityError("checkpoint configuration fingerprint does not match")

    state_name = manifest.get("state_file")
    if not isinstance(state_name, str) or _STATE_NAME.fullmatch(state_name) is None:
        raise CheckpointIntegrityError("checkpoint state filename is invalid")
    try:
        payload = (directory / state_name).read_bytes()
    except OSError as error:
        raise CheckpointIntegrityError("checkpoint state file cannot be read") from error

    expected_size = manifest.get("state_bytes")
    expected_checksum = manifest.get("state_sha256")
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if expected_size != len(payload) or expected_checksum != actual_checksum:
        raise CheckpointIntegrityError("checkpoint state checksum or byte count does not match")

    if target_state is None:
        key = jax.random.PRNGKey(config.seed)
        target_state = make_initialize(config)(key)
    try:
        restored = serialization.from_bytes(target_state, payload)
    except Exception as error:
        raise CheckpointCompatibilityError("checkpoint state structure is incompatible") from error

    restored_updates = int(np.asarray(restored.completed_updates))
    if manifest.get("completed_updates") != restored_updates:
        raise CheckpointIntegrityError("checkpoint update counter does not match its manifest")
    return restored

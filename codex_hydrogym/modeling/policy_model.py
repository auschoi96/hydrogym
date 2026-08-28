"""Models-from-Code MLflow pyfunc for a deterministic codex_hydrogym policy."""

import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pandas as pd

from codex_hydrogym import PROJECT_LABEL
from codex_hydrogym.config import KolmogorovPPOConfig, config_fingerprint
from codex_hydrogym.modeling.network import ActorCritic
from codex_hydrogym.modeling.schema import (
    ACTION_COLUMNS,
    ACTION_DIM,
    ACTION_MAXIMUM,
    ACTION_MINIMUM,
    observation_columns,
)


POLICY_ASSET_FORMAT = "codex_hydrogym.flax_policy.v1"


def _read_json(path: str | Path, label: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read codex_hydrogym {label}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"codex_hydrogym {label} must be a JSON object")
    return payload


class CodexHydrogymPolicyModel(mlflow.pyfunc.PythonModel):
    """Serve the clipped policy mean; no random key or sampling path exists."""

    def load_context(self, context) -> None:
        config_payload = _read_json(context.artifacts["config"], "config")
        if config_payload.get("project_label") != PROJECT_LABEL:
            raise RuntimeError("MLflow config artifact has the wrong project label")
        raw_config = config_payload.get("config")
        if not isinstance(raw_config, dict):
            raise RuntimeError("MLflow config artifact is missing its config object")
        raw_config = dict(raw_config)
        raw_config.pop("project_label", None)
        self.config = KolmogorovPPOConfig(**raw_config)

        validation = _read_json(context.artifacts["physics_validation"], "physics validation")
        if validation.get("project_label") != PROJECT_LABEL or validation.get("passed") is not True:
            raise RuntimeError("refusing to load a controller without passing physics validation")

        policy_directory = Path(context.artifacts["policy"])
        manifest = _read_json(policy_directory / "manifest.json", "policy manifest")
        if manifest.get("format") != POLICY_ASSET_FORMAT or manifest.get("project_label") != PROJECT_LABEL:
            raise RuntimeError("MLflow policy artifact format is incompatible")
        if manifest.get("config_fingerprint") != config_fingerprint(self.config):
            raise RuntimeError("MLflow policy and config fingerprints do not match")
        if manifest.get("physics_validation_passed") is not True:
            raise RuntimeError("MLflow policy manifest does not contain a passing physics gate")

        self.observation_dimension = self.config.obs_size**2
        if manifest.get("observation_dimension") != self.observation_dimension:
            raise RuntimeError("MLflow policy observation dimension does not match its config")
        if manifest.get("action_dimension") != ACTION_DIM:
            raise RuntimeError("MLflow policy action dimension is incompatible")

        params_name = manifest.get("params_file")
        if params_name != "params.msgpack":
            raise RuntimeError("MLflow policy parameter filename is invalid")
        try:
            payload = (policy_directory / params_name).read_bytes()
        except OSError as error:
            raise RuntimeError("cannot read MLflow policy parameters") from error
        if len(payload) != manifest.get("params_bytes"):
            raise RuntimeError("MLflow policy parameter byte count does not match")
        if hashlib.sha256(payload).hexdigest() != manifest.get("params_sha256"):
            raise RuntimeError("MLflow policy parameter checksum does not match")

        if self.config.precision == "float64":
            jax.config.update("jax_enable_x64", True)
            self.numpy_dtype = np.float64
            jax_dtype = jnp.float64
        else:
            self.numpy_dtype = np.float32
            jax_dtype = jnp.float32

        network = ActorCritic(ACTION_DIM, activation=self.config.activation)
        template = network.init(
            jax.random.PRNGKey(0),
            jnp.zeros((self.observation_dimension,), dtype=jax_dtype),
        )
        try:
            parameters = serialization.from_bytes(template, payload)
        except Exception as error:
            raise RuntimeError("MLflow policy parameters are structurally incompatible") from error

        mean = np.asarray(manifest.get("observation_mean"), dtype=self.numpy_dtype)
        variance = np.asarray(manifest.get("observation_variance"), dtype=self.numpy_dtype)
        expected_shape = (self.observation_dimension,)
        if mean.shape != expected_shape or variance.shape != expected_shape:
            raise RuntimeError("MLflow observation normalization shape is incompatible")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise RuntimeError("MLflow observation normalization values are invalid")
        self.observation_mean = mean
        self.observation_variance = variance
        self.normalization_epsilon = float(manifest.get("normalization_epsilon", 1.0e-8))
        self.expected_columns = observation_columns(self.observation_dimension)

        def deterministic_actions(observations):
            policy, _value = network.apply(parameters, observations)
            return jnp.clip(policy.mode(), ACTION_MINIMUM, ACTION_MAXIMUM)

        self._deterministic_actions = jax.jit(deterministic_actions)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        if params:
            raise ValueError("codex_hydrogym deterministic inference accepts no runtime parameters")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("codex_hydrogym policy input must be a pandas DataFrame")
        missing = sorted(set(self.expected_columns) - set(model_input.columns))
        unknown = sorted(set(model_input.columns) - set(self.expected_columns))
        if missing or unknown:
            raise ValueError(
                f"observation columns do not match the policy signature; missing={missing}, unknown={unknown}"
            )
        observations = model_input.loc[:, self.expected_columns].to_numpy(dtype=self.numpy_dtype, copy=True)
        if observations.ndim != 2 or observations.shape[0] == 0:
            raise ValueError("codex_hydrogym policy input must contain at least one observation row")
        if not np.all(np.isfinite(observations)):
            raise ValueError("codex_hydrogym policy observations must be finite")

        normalized = (observations - self.observation_mean) / np.sqrt(
            self.observation_variance + self.normalization_epsilon
        )
        actions = np.asarray(jax.device_get(self._deterministic_actions(jnp.asarray(normalized))))
        if actions.shape != (len(model_input), ACTION_DIM) or not np.all(np.isfinite(actions)):
            raise RuntimeError("codex_hydrogym policy produced invalid actions")
        return pd.DataFrame(actions, columns=ACTION_COLUMNS, index=model_input.index)


mlflow.models.set_model(CodexHydrogymPolicyModel())

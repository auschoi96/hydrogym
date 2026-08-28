"""Export a compact HydroGym JAX reference trajectory for the AppKit cockpit."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


PROJECT_LABEL = "codex_hydrogym"
GRID_SIZE = (32, 32)
DT = 0.01
ACTION_TIME = 0.2
SAVE_TIME = 0.02
SEED = 20260822


def main() -> None:
    environment = KolmogorovFlow(
        env_config={
            "dt": DT,
            "action_time": ACTION_TIME,
            "save_time": SAVE_TIME,
            "reward_alpha": 1.0,
            "initial_perturbation_amplitude": 1.0e-3,
            "max_episode_steps": 8,
        },
        flow_config={
            "Re": 200.0,
            "k": 4,
            "grid_size": GRID_SIZE,
            "obs_size": 8,
        },
    )
    params = environment.default_params
    key = jax.random.PRNGKey(SEED)
    key, reset_key = jax.random.split(key)
    _, state = environment.reset_env(reset_key, params)

    spectral_frames = [np.asarray(frame) for frame in np.asarray(state.trajectory)]
    zero_action = jnp.zeros((params.action_dim,), dtype=jnp.float32)
    for _ in range(3):
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = environment.step_env(step_key, state, zero_action, params)
        spectral_frames.extend(np.asarray(frame) for frame in np.asarray(state.trajectory))

    sample_indices = np.linspace(0, len(spectral_frames) - 1, num=min(24, len(spectral_frames)), dtype=int)
    fields = [
        np.fft.irfftn(spectral_frames[index], s=GRID_SIZE, axes=(-2, -1)).real
        for index in sample_indices
    ]
    maximum_abs_vorticity = max(float(np.max(np.abs(field))) for field in fields)
    frames = [
        {
            "time": round(frame_index * SAVE_TIME, 6),
            "values": np.round(field, 6).reshape(-1).tolist(),
        }
        for frame_index, field in enumerate(fields)
    ]

    output = Path(__file__).resolve().parents[1] / "client" / "public" / f"{PROJECT_LABEL}_reference_flow.json"
    output.write_text(
        json.dumps(
            {
                "formatVersion": 1,
                "projectLabel": PROJECT_LABEL,
                "datasetKind": "reference_simulation",
                "evidenceStatus": "not_ppo_training",
                "source": "HydroGym JAX KolmogorovFlow pseudo-spectral solver",
                "description": "Uncontrolled zero-action baseline generated from the repository solver.",
                "generatedAt": "2026-08-22T00:00:00Z",
                "seed": SEED,
                "reynoldsNumber": 200.0,
                "forcingWavenumber": 4,
                "gridSize": list(GRID_SIZE),
                "observationGrid": [8, 8],
                "actionDimension": int(params.action_dim),
                "action": [0.0] * int(params.action_dim),
                "dt": DT,
                "actionTime": ACTION_TIME,
                "saveTime": SAVE_TIME,
                "maximumAbsVorticity": maximum_abs_vorticity,
                "frames": frames,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(frames)} {PROJECT_LABEL} reference frames to {output}")


if __name__ == "__main__":
    main()

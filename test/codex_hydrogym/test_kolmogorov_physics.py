"""Physics contracts for the codex_hydrogym Kolmogorov demo."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from hydrogym.jax.envs.kolmogorov import FlowConfig, KolmogorovFlow, KolmogorovFlowState, PseudoSpectralNavierStokes2D
from hydrogym.jax.kolmogorov_contract import FORCED_MODE_QUADRATURE_BASIS_VERSION
from hydrogym.jax.utils.utils import dealiasing
from hydrogym.jax.kolmogorov_contract import (
    LEGACY_SPEED_GRID_OBSERVATION_MODE,
    SIGNED_FORCED_MODE_OBSERVATION_MODE,
    SIGNED_FORCED_MODE_OBSERVATION_VERSION,
)


def test_real_mesh_is_periodic_without_a_duplicate_endpoint_and_matches_fft_mesh():
    grid_size = (8, 10)
    domain_x = (1.0, 1.0 + 2.0 * np.pi)
    domain_y = (-np.pi, np.pi)
    flow = FlowConfig(grid_size=grid_size, domain_x=domain_x, domain_y=domain_y)

    x, y = (np.asarray(value) for value in flow.load_mesh("codex_hydrogym"))
    kx, ky = (np.asarray(value) for value in flow.load_fft_mesh())

    assert x.shape == grid_size
    assert y.shape == grid_size
    assert x[-1, 0] < domain_x[1]
    assert y[0, -1] < domain_y[1]
    np.testing.assert_allclose(
        np.diff(x[:, 0]),
        (domain_x[1] - domain_x[0]) / grid_size[0],
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.diff(y[0, :]),
        (domain_y[1] - domain_y[0]) / grid_size[1],
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    expected_kx = np.fft.fftfreq(grid_size[0], d=(domain_x[1] - domain_x[0]) / grid_size[0])
    expected_ky = np.fft.rfftfreq(grid_size[1], d=(domain_y[1] - domain_y[0]) / grid_size[1])
    np.testing.assert_allclose(kx[:, 0], expected_kx, rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(ky[0, :], expected_ky, rtol=1.0e-6, atol=1.0e-7)


def test_default_forcing_is_monochromatic():
    flow = FlowConfig()
    _, y = flow.load_mesh("codex_hydrogym")
    forcing_x, _ = flow.forcing_function(flow.k, flow.load_mesh("codex_hydrogym")[0], y)

    power = np.abs(np.fft.rfft(np.asarray(forcing_x[0]))) ** 2
    peak = int(np.argmax(power))
    non_peak_fraction = 1.0 - power[peak] / power.sum()

    assert peak == flow.k
    assert non_peak_fraction < 1.0e-10


def test_forcing_phase_is_explicit_and_default_preserves_the_original_field():
    default_flow = FlowConfig(grid_size=(8, 32))
    phased_flow = FlowConfig(grid_size=(8, 32), forcing_phase=np.pi / 2.0)
    x, y = default_flow.load_mesh("codex_hydrogym")

    default_x, default_y = default_flow.forcing_function(default_flow.k, x, y)
    phased_x, phased_y = phased_flow.forcing_function(phased_flow.k, x, y)

    np.testing.assert_allclose(default_x, np.sin(default_flow.k * np.asarray(y)), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        phased_x,
        np.sin(phased_flow.k * np.asarray(y) + np.pi / 2.0),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(default_y, np.zeros_like(default_y))
    np.testing.assert_array_equal(phased_y, np.zeros_like(phased_y))


def test_forced_mode_actuators_form_a_versioned_sine_cosine_quadrature_pair():
    env = KolmogorovFlow(flow_config={"grid_size": (8, 32), "obs_size": 4, "k": 4})
    params = env.default_params
    unit_actions = np.eye(params.action_dim, dtype=np.float32)

    fields = [np.asarray(env._control_field(jnp.asarray(action), params)[0]) for action in unit_actions]
    expected = (
        np.sin(env.flow.k * np.asarray(env.y)),
        np.cos(env.flow.k * np.asarray(env.y)),
        np.sin(params.k3 * np.asarray(env.y)),
        np.sin(params.k4 * np.asarray(env.y)),
    )
    for actual, target in zip(fields, expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=1.0e-6, atol=1.0e-6)

    metadata = env.action_basis_metadata(params)
    assert metadata["version"] == FORCED_MODE_QUADRATURE_BASIS_VERSION
    assert metadata["channels"][0]["wavenumber"] == env.flow.k
    assert metadata["channels"][1]["wavenumber"] == env.flow.k
    assert metadata["channels"][0]["function"] == "sin"
    assert metadata["channels"][1]["function"] == "cos"


def test_quadrature_coefficients_span_every_forcing_phase():
    env = KolmogorovFlow(flow_config={"grid_size": (8, 32), "obs_size": 4, "k": 4})
    params = env.default_params
    amplitude = float(params.max_action)

    for phase in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 0.37):
        action = jnp.array([amplitude * np.cos(phase), amplitude * np.sin(phase), 0.0, 0.0])
        actual, _ = env._control_field(action, params)
        expected = amplitude * np.sin(env.flow.k * np.asarray(env.y) + phase)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_signed_forced_mode_observation_is_opt_in_versioned_and_amplitude_preserving():
    legacy = KolmogorovFlow(flow_config={"grid_size": (8, 32), "obs_size": 4, "k": 4})
    signed = KolmogorovFlow(
        env_config={"observation_mode": SIGNED_FORCED_MODE_OBSERVATION_MODE},
        flow_config={"grid_size": (8, 32), "obs_size": 4, "k": 4},
    )
    sine_amplitude, cosine_amplitude = 0.3, -0.2
    # For an x-independent velocity field, omega = -du/dy.
    vorticity = -signed.flow.k * (
        sine_amplitude * jnp.cos(signed.flow.k * signed.y)
        - cosine_amplitude * jnp.sin(signed.flow.k * signed.y)
    )
    trajectory = jnp.fft.rfftn(vorticity)[jnp.newaxis, ...]

    observation = signed._trajectory_signed_forced_mode_obs(trajectory)

    np.testing.assert_allclose(observation, [sine_amplitude, cosine_amplitude], rtol=1.0e-5, atol=1.0e-5)
    assert legacy.default_params.obs_dim == legacy.flow.obs_size**2
    assert legacy.observation_metadata()["mode"] == LEGACY_SPEED_GRID_OBSERVATION_MODE
    assert signed.default_params.obs_dim == 2
    assert signed.observation_metadata()["version"] == SIGNED_FORCED_MODE_OBSERVATION_VERSION


def test_control_vorticity_is_the_analytic_curl_of_the_vector_field():
    flow = FlowConfig(grid_size=(24, 28))
    equation = PseudoSpectralNavierStokes2D(flow)
    x, y = flow.load_mesh("codex_hydrogym")
    control_x = jnp.sin(3.0 * y)
    control_y = jnp.cos(2.0 * x)
    omega_hat = jnp.zeros_like(jnp.fft.rfftn(control_x))

    actual_hat = equation.control_term(omega_hat, (control_x, control_y))
    actual = jnp.fft.irfftn(actual_hat, s=flow.grid_size)
    expected = -2.0 * jnp.sin(2.0 * x) - 3.0 * jnp.cos(3.0 * y)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_initial_vorticity_matches_documented_streamfunction():
    flow = FlowConfig()
    x, y = flow.load_mesh("codex_hydrogym")

    actual = np.asarray(jnp.fft.irfftn(flow.initialize_state(), s=flow.grid_size))
    expected = 2.0 * np.sin(np.asarray(x)) * np.cos(np.asarray(y))

    relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_l2 < 1.0e-5


def test_dealiasing_removes_modes_outside_the_two_thirds_band():
    spectral_field = jnp.ones((12, 7), dtype=jnp.complex64)

    filtered = np.asarray(dealiasing(spectral_field))

    assert filtered[0, 0] == 1.0
    assert filtered[-1, 0] == 1.0
    assert filtered[5, 0] == 0.0
    assert filtered[0, 5] == 0.0


def test_default_reward_includes_the_physics_term():
    env = KolmogorovFlow()

    assert env.default_params.reward_alpha > 0.0


def test_reward_is_decomposed_into_physics_and_control_terms():
    env = KolmogorovFlow(flow_config={"grid_size": (8, 8), "obs_size": 4})
    params = env.default_params.replace(reward_alpha=2.0)
    trajectory = env.flow.initial_vorticity()[jnp.newaxis, ...]
    action = jnp.array([0.1, -0.2, 0.3, -0.4])

    terms = env._reward_terms(action, trajectory, params)
    reward = env._reward(action, trajectory, params)

    np.testing.assert_allclose(np.asarray(reward), np.asarray(terms["tke"] + terms["action_l1"]))
    np.testing.assert_allclose(np.asarray(terms["action_l1"]), -1.0)
    assert float(terms["tke"]) < 0.0


def test_step_info_exposes_control_effort_and_action_slew():
    env = KolmogorovFlow(flow_config={"grid_size": (8, 8), "obs_size": 4})
    params = env.default_params
    omega = env.flow.initial_vorticity()
    trajectory = omega[jnp.newaxis, ...]
    previous_action = jnp.array([0.0, 0.1, 0.2, 0.3])
    state = KolmogorovFlowState(
        trajectory=trajectory,
        omega_hat=omega,
        time=jnp.array(0),
        terminal=jnp.array(False),
        last_action=previous_action,
    )
    action = jnp.array([0.1, 0.1, 0.0, 0.3])
    env._rollout = lambda omega_hat0, params, control_field=None: (omega_hat0, trajectory)

    _, next_state, reward, _, info = env.step_env(jax.random.PRNGKey(0), state, action, params)

    np.testing.assert_allclose(np.asarray(info["control_l1"]), np.sum(np.abs(np.asarray(action))))
    np.testing.assert_allclose(np.asarray(info["control_l2"]), np.sum(np.asarray(action) ** 2))
    np.testing.assert_allclose(
        np.asarray(info["action_delta_l2"]),
        np.sum((np.asarray(action) - np.asarray(previous_action)) ** 2),
    )
    np.testing.assert_allclose(np.asarray(info["reward_total"]), np.asarray(reward))
    np.testing.assert_allclose(np.asarray(next_state.last_action), np.asarray(action))

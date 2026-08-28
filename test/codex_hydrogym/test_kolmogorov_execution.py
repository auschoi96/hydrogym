"""Execution and reproducibility contracts for codex_hydrogym."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hydrogym.jax.envs.kolmogorov import FlowConfig, KolmogorovFlow
from hydrogym.jax.solvers import base as solver_module
from hydrogym.jax.solvers.base import RungeKuttaCrankNicolson


class _ConstantEquation:
    def nonlinear_terms(self, state, control_field=None):
        return jnp.ones_like(state)

    def linear_terms(self, state):
        return jnp.zeros_like(state)

    def implicit_timestep(self, state, timestep):
        return state


class _DummyFlow:
    def initialize_state(self):
        return jnp.array(0.0)


def test_step_counts_use_exact_ratios_instead_of_float_floor_division():
    assert solver_module._exact_step_count(1.0, 1.0e-3, "save interval") == 1_000
    assert solver_module._exact_step_count(10.0, 1.0e-3, "action interval") == 10_000

    with pytest.raises(ValueError, match="integer multiple"):
        solver_module._exact_step_count(1.0, 0.3, "invalid interval")


def test_runge_kutta_step_uses_the_timestep_passed_to_solve():
    flow = _DummyFlow()
    solver = RungeKuttaCrankNicolson(flow, dt=0.1, save_n=1, equation=_ConstantEquation())

    advance = solver.step(flow, dt=0.25, save_n=1, callbacks=[])

    np.testing.assert_allclose(np.asarray(advance(jnp.array(0.0))), 0.25, rtol=1.0e-6)


def test_kolmogorov_rollout_honors_static_timing_parameters():
    env = KolmogorovFlow(flow_config={"grid_size": (8, 8), "obs_size": 4})
    captured = {}

    class _RecordingIntegrator:
        def solve(self, **kwargs):
            captured.update(kwargs)
            state = kwargs["initial_state"]
            return state, state[jnp.newaxis, ...]

    env.integrator = _RecordingIntegrator()
    params = env.default_params.replace(dt=0.25, action_time=2.0, save_time=0.5)
    omega = env.flow.initialize_state()

    env._rollout(omega, params)

    assert captured["dt"] == 0.25
    assert captured["t_span"] == (0.0, 2.0)
    assert captured["save_n"] == 0.5


def test_seeded_streamfunction_perturbations_are_repeatable_and_distinct():
    flow = FlowConfig(grid_size=(8, 8))
    amplitude = 1.0e-3

    first = flow.initial_vorticity(jax.random.PRNGKey(7), perturbation_amplitude=amplitude)
    repeated = flow.initial_vorticity(jax.random.PRNGKey(7), perturbation_amplitude=amplitude)
    different = flow.initial_vorticity(jax.random.PRNGKey(8), perturbation_amplitude=amplitude)

    np.testing.assert_allclose(np.asarray(first), np.asarray(repeated))
    assert not np.allclose(np.asarray(first), np.asarray(different))


def test_solver_traces_one_inner_advance_and_does_not_mutate_flow_state():
    flow = _DummyFlow()
    solver = object.__new__(RungeKuttaCrankNicolson)
    trace_calls = []

    def make_step(flow, dt, save_n, callbacks, control_field=None):
        def advance(state):
            trace_calls.append(None)
            return state + float(save_n)

        return advance

    solver.step = make_step
    final_state, saved_states = solver.solve(
        dt=0.25,
        flow=flow,
        t_span=(0.0, 1.0),
        save_n=0.5,
        initial_state=jnp.array(0.0),
    )

    np.testing.assert_allclose(np.asarray(final_state), 4.0)
    np.testing.assert_allclose(np.asarray(saved_states), np.array([2.0, 4.0]))
    assert len(trace_calls) == 1
    assert not hasattr(flow, "vorticity")


def test_solver_accepts_exact_subunit_action_intervals():
    flow = _DummyFlow()
    solver = RungeKuttaCrankNicolson(flow, dt=0.05, save_n=1, equation=_ConstantEquation())

    final_state, saved_states = solver.solve(
        dt=0.05,
        flow=flow,
        t_span=(0.0, 0.1),
        save_n=0.05,
        initial_state=jnp.array(0.0),
    )

    np.testing.assert_allclose(np.asarray(final_state), 0.1, rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(saved_states), np.array([0.05, 0.1]), rtol=1.0e-6)


def test_jitted_reset_and_step_are_finite_and_keyed():
    env = KolmogorovFlow(
        env_config={
            "dt": 0.05,
            "action_time": 1.0,
            "save_time": 0.5,
            "initial_perturbation_amplitude": 1.0e-3,
        },
        flow_config={"grid_size": (16, 16), "obs_size": 4},
    )
    params = env.default_params
    reset = jax.jit(env.reset_env)
    step = jax.jit(env.step_env)

    obs_a, state_a = reset(jax.random.PRNGKey(11), params)
    obs_a_repeat, state_a_repeat = reset(jax.random.PRNGKey(11), params)
    _, state_b = reset(jax.random.PRNGKey(12), params)

    np.testing.assert_allclose(np.asarray(obs_a), np.asarray(obs_a_repeat))
    np.testing.assert_allclose(np.asarray(state_a.omega_hat), np.asarray(state_a_repeat.omega_hat))
    assert not np.allclose(np.asarray(state_a.omega_hat), np.asarray(state_b.omega_hat))
    assert obs_a.shape == (16,)
    assert state_a.trajectory.shape == (2, 16, 9)
    assert np.isfinite(np.asarray(obs_a)).all()
    assert np.isfinite(np.asarray(state_a.trajectory)).all()

    next_obs, next_state, reward, done, info = step(
        jax.random.PRNGKey(13),
        state_a,
        jnp.zeros((4,)),
        params,
    )

    assert np.isfinite(np.asarray(next_obs)).all()
    assert np.isfinite(np.asarray(next_state.trajectory)).all()
    assert np.isfinite(np.asarray(reward))
    assert not bool(np.asarray(done))
    assert np.isfinite(np.asarray(info["mean_tke"]))

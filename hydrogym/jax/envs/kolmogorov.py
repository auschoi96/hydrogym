import math
from typing import Callable, Dict, Iterable, NamedTuple, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import tree_math
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax

from hydrogym.core import CallbackBase, PDEBase, TransientSolver
from hydrogym.jax.env_core import EnvParams, JAXFlowEnvBase
from hydrogym.jax.equation import IMEXEquation
from hydrogym.jax.kolmogorov_contract import (
    FORCED_MODE_QUADRATURE_BASIS_VERSION,
    LEGACY_SPEED_GRID_OBSERVATION_MODE,
    OBSERVATION_CONTRACT_VERSIONS,
    SIGNED_FORCED_MODE_OBSERVATION_MODE,
)
from hydrogym.jax.solvers.base import RungeKuttaCrankNicolson
from hydrogym.jax.utils.utils import compute_real_velocity_point, compute_tke, compute_velocity_fft, dealiasing


#######################################################################################
#                                                                                     #
#                             FLOW CONFIGURATION                                      #
#                                                                                     #
#######################################################################################


class FlowConfig(PDEBase):
    DEFAULT_REYNOLDS = 200
    DEFAULT_WAVENUMBER = 4
    DEFAULT_GRID_SIZE = (64, 64)
    DEFAULT_DOMAIN_X = (0, 2 * jnp.pi)
    DEFAULT_DOMAIN_Y = (0, 2 * jnp.pi)
    DEFAULT_OBS_SIZE = 8  # This correlates to a total observation size of 8x8 = 64.
    DEFAULT_FORCING_PHASE = 0.0

    def __init__(self, **config):
        self.k = config.get("k", self.DEFAULT_WAVENUMBER)
        self.Re = config.get("Re", self.DEFAULT_REYNOLDS)
        self.grid_size = config.get("grid_size", self.DEFAULT_GRID_SIZE)
        self.domain_x = config.get("domain_x", self.DEFAULT_DOMAIN_X)
        self.domain_y = config.get("domain_y", self.DEFAULT_DOMAIN_Y)
        self.obs_size = config.get("obs_size", self.DEFAULT_OBS_SIZE)
        self.forcing_phase = float(config.get("forcing_phase", self.DEFAULT_FORCING_PHASE))
        if not math.isfinite(self.forcing_phase):
            raise ValueError("forcing_phase must be finite")
        self.control_function = (
            jnp.zeros_like(self.load_mesh("default")[0]),
            jnp.zeros_like(self.load_mesh("default")[1]),
        )

        super().__init__(**config)

    def load_mesh(self, name):
        """
        Create jax grid given the desired dimensions and spacing in real space

        Returns:
            jax meshgrid
        """
        x0, xn, nx = self.domain_x[0], self.domain_x[1], self.grid_size[0]
        y0, yn, ny = self.domain_y[0], self.domain_y[1], self.grid_size[1]
        # A periodic pseudospectral grid contains the lower endpoint but not a
        # duplicate copy of the upper endpoint.
        x = jnp.linspace(x0, xn, nx, endpoint=False)
        y = jnp.linspace(y0, yn, ny, endpoint=False)
        return jnp.meshgrid(x, y, indexing="ij")

    def _calculate_velocity_point(self, state, k1, k2):
        # Calculate velocity point
        kx, ky = self.load_fft_mesh()
        uhat, vhat = compute_velocity_fft(state, kx, ky)
        point_velocity = compute_real_velocity_point(uhat, vhat, k1, k2)

        return point_velocity

    def state(self) -> jnp.array:
        return self.state

    def get_observations(self) -> jnp.array:
        n, m = self.grid_size
        divisor = n // self.obs_size

        def calculate_velocity(trajectory):
            points = [
                self._calculate_velocity_point(trajectory, x, y)
                for x in range(0, n, int(n / divisor))
                for y in range(0, m, int(m / divisor))
            ]
            return jnp.array(points)

        def scan_fn(carry, state):
            obs_val = calculate_velocity(state)  # To use energy observation, swap this with calculate_energy
            return carry, obs_val

        _, all_obs = lax.scan(scan_fn, None, self.vorticity)
        return all_obs

    def load_fft_mesh(self):
        """Create jax grid given desired dimensions and spacing in real Fourier space

        Returns:
            jax meshgrid
        """
        N = self.grid_size[0]
        M = self.grid_size[1]
        dx = (self.domain_x[1] - self.domain_x[0]) / N
        dy = (self.domain_y[1] - self.domain_y[0]) / M
        kx = jnp.fft.fftfreq(N, dx)
        ky = jnp.fft.rfftfreq(M, dy)
        return jnp.meshgrid(kx, ky, indexing="ij")

    def initial_vorticity(
        self,
        key: Optional[chex.PRNGKey] = None,
        perturbation_amplitude: float = 0.0,
    ) -> jnp.ndarray:
        """Build a reproducible divergence-free initial vorticity field.

        ``perturbation_amplitude`` is the amplitude of an additional
        streamfunction mode. A zero value preserves the deterministic base
        state; a nonzero value uses ``key`` to randomize that mode's phases.
        """
        x, y = self.load_mesh("default")
        base_vorticity = 2.0 * jnp.sin(x) * jnp.cos(y)

        phases = (
            jnp.zeros((2,)) if key is None else jax.random.uniform(key, shape=(2,), minval=0.0, maxval=2.0 * jnp.pi)
        )
        mode_x, mode_y = 2, 3
        perturbation_vorticity = (
            (mode_x**2 + mode_y**2)
            * perturbation_amplitude
            * jnp.sin(mode_x * x + phases[0])
            * jnp.cos(mode_y * y + phases[1])
        )
        return jnp.fft.rfftn(base_vorticity + perturbation_vorticity)

    def initialize_state(self):
        """Generate a divergence free velocity field to initialize the state
        Initializing with divergence free field specified with the following stream function:

        φ(x,y) = sin(x)cos(y)

        Returns:
            fft vorticity field
        """
        self.vorticity = self.initial_vorticity()
        return self.vorticity

    def set_BCs(self):
        # Set the boundary conditions
        pass

    def forcing_function(self, k, x, y):
        """Sinusoidal forcing function that drives the Kolmogorov flow.

        Args:
            k (int): forcing wavenumber
            x (jnp.array): spatial coordinates in x
            y (jnp.array): spatial coordinates in y

        Returns:
            tuple: forcing function in (x,y)
        """
        return (jnp.sin(k * y + self.forcing_phase), jnp.zeros_like(y))

    def evaluate_objective(self):
        """Return a copy of the flow state"""
        pass

    @property
    def nu(self):
        return 1 / self.Re

    @property
    def num_inputs(self) -> int:
        """Length of the control vector (number of actuators)"""
        return 2

    @property
    def num_outputs(self) -> int:
        """Number of scalar observed variables"""
        pass

    def save_checkpoint(self):
        """Set up mesh, function spaces, state vector, etc"""
        pass

    def init_bcs(self):
        """Initialize any boundary conditions for the PDE."""
        pass

    def copy_state(self, deepcopy=True):
        """Return a copy of the flow state"""
        pass

    def render(self, **kwargs):
        """Plot the current PDE state (called by `gym.Env`)"""
        pass

    def load_checkpoint(self, filename: str):
        pass


#######################################################################################
#                                                                                     #
#                             PSEUDOSPECTRAL EQUATION                                 #
#                                                                                     #
#######################################################################################


class PseudoSpectralNavierStokes2D(IMEXEquation):
    """
    Calculates the 2D Navier-Stokes equations using the pseudo-spectral solver.
    We transform the 2D Navier-Stokes equation to a vorticity equation:
        ∂/∂t ω + u·∇ω = v ∇²ω + ƒ ;
        ω = - ∇²φ ;
    and solve in Fourier space
    """

    def __init__(self, flow: FlowConfig):
        self.flow = flow
        self.grid = flow.load_fft_mesh()
        self.real_grid = flow.load_mesh("name")
        self.kx, self.ky = self.grid
        self.x, self.y = self.real_grid

    def linear_terms(self, omega_hat):
        """Computes the linear (viscous) term of the vorticity equation"""
        return self.flow.nu * (2j * jnp.pi) ** 2 * (self.kx**2 + self.ky**2) * omega_hat

    def implicit_timestep(self, omega_hat, time_step):
        """
        Function that computes an implicit euler timestep,
          y_n+1 = y_n / (1-∇tλ).

        """
        double_derivative = (2j * jnp.pi) ** 2 * (self.kx**2 + self.ky**2)
        return 1 / (1 - time_step * self.flow.nu * double_derivative) * omega_hat

    def nonlinear_terms(self, omega_hat, control_field=None):
        """Computes the explicit (nonlinear) terms in the vorticity equation.
        Uses the stream function to compute velocity components in Fourier space.

        Args:
            omega_hat: fft of vorticity
            control_field: tuple (cfx, cfy) of physical-space forcing arrays, or None

        Returns:
            terms: Nonlinear terms of the equation.
        """

        kx, ky = self.kx, self.ky

        double_derivative = (2 * jnp.pi * 1j) ** 2 * (abs(self.kx) ** 2 + abs(ky) ** 2)
        double_derivative = double_derivative.at[0, 0].set(1)  # avoiding division by 0.0 in the next step

        psi_hat = -1 * omega_hat / double_derivative
        uhat = (2 * jnp.pi * 1j) * ky * psi_hat  # Get u,v from phi
        vhat = (-1 * 2 * jnp.pi * 1j) * kx * psi_hat

        u, v = jnp.fft.irfftn(uhat), jnp.fft.irfftn(vhat)

        grad_x_hat = 2j * jnp.pi * self.kx * omega_hat
        grad_y_hat = 2j * jnp.pi * self.ky * omega_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * u + grad_y * v)
        advection_hat = jnp.fft.rfftn(advection)

        forcing_hat = self.forcing_term()
        control_hat = self.control_term(omega_hat, control_field=control_field)
        advection_hat = dealiasing(advection_hat)  # 2/3 dealiasing rule

        terms = advection_hat + forcing_hat + control_hat
        return terms

    def control_term(self, omega_hat, control_field=None):
        """Computes the user-specified forcing term of the vorticity equation
        Args:
          omega_hat: Fourier transformed vorticity term
          control_field: tuple (cfx, cfy) of physical-space forcing arrays, or None
        """
        if control_field is None:
            return jnp.zeros_like(omega_hat)

        cfx, cfy = control_field
        cfx_hat = jnp.fft.rfftn(cfx)
        cfy_hat = jnp.fft.rfftn(cfy)

        # ``fftfreq`` is measured in cycles per unit length, so the Fourier
        # symbol for a physical derivative is 2*pi*i*k.  The vorticity source
        # is the z-component of curl(control): d(cfy)/dx - d(cfx)/dy.
        return 2j * jnp.pi * (self.kx * cfy_hat - self.ky * cfx_hat)

    def forcing_term(self):
        """Computes the user-specified forcing term of the vorticity equation
        Args:
          omega_hat: Fourier transformed vorticity term
          forcing: Forcing function as specified by environment or user
        """
        forcing_func = self.flow.forcing_function
        if forcing_func is not None:
            kx, ky = self.grid
            x, y = self.real_grid
            fx, fy = forcing_func(k=self.flow.k, x=x, y=y)
            fx_hat, fy_hat = jnp.fft.rfft2(fx), jnp.fft.rfft2(fy)

            # Transform the velocity forcing into vorticity
            derivative_term = 2j * jnp.pi
            f_vorticity = derivative_term * (fy_hat * kx - fx_hat * ky)
            return f_vorticity
        else:
            return None


#######################################################################################
#                                                                                     #
#                             GYMNAX ENVIRONMENT                                      #
#                                                                                     #
#######################################################################################


@struct.dataclass
class KolmogorovFlowState(environment.EnvState):
    trajectory: jnp.ndarray
    omega_hat: jnp.ndarray
    time: jnp.ndarray
    terminal: jnp.ndarray
    last_action: jnp.ndarray


@struct.dataclass
class KolmogorovFlowParams(EnvParams):
    min_action: float = -0.5
    max_action: float = 0.5
    min_obs: float = -jnp.inf
    max_obs: float = jnp.inf

    # These values determine lax.scan lengths and are therefore static PyTree
    # metadata. Changing one intentionally produces a new JAX compilation.
    dt: float = struct.field(pytree_node=False, default=1e-3)
    action_time: float = struct.field(pytree_node=False, default=10.0)
    save_time: float = struct.field(pytree_node=False, default=1.0)

    # Channels 0 and 1 are the sine/cosine quadrature pair at the forced
    # wavenumber. Keeping channels 2 and 3 at their legacy modes preserves
    # three of the four original action meanings while retaining a 4-D API.
    k1: int = 4
    k2: int = 4
    k3: int = 6
    k4: int = 7

    action_dim: int = struct.field(pytree_node=False, default=4)
    obs_dim: int = struct.field(pytree_node=False, default=64)
    max_episode_steps: int = 1000
    reward_alpha: float = 1.0
    initial_perturbation_amplitude: float = 0.0

    include_grad: bool = True


class KolmogorovFlow(JAXFlowEnvBase):
    def __init__(
        self,
        env_config: Optional[Dict] = None,
        flow_config: Optional[Dict] = None,
    ):
        super().__init__(env_config)

        self.observation_mode = self.env_config.get(
            "observation_mode",
            LEGACY_SPEED_GRID_OBSERVATION_MODE,
        )
        if self.observation_mode not in OBSERVATION_CONTRACT_VERSIONS:
            raise ValueError(
                f"observation_mode must be one of {sorted(OBSERVATION_CONTRACT_VERSIONS)}"
            )

        self.flow = FlowConfig(**(flow_config or {}))

        self.n, self.m = self.flow.grid_size
        self.x, self.y = self.flow.load_mesh("")
        self.kx, self.ky = self.flow.load_fft_mesh()

        default = self.default_params
        self.equation = PseudoSpectralNavierStokes2D(self.flow)
        self.integrator = RungeKuttaCrankNicolson(
            flow=self.flow,
            dt=float(default.dt),
            save_n=int(default.save_time),
            equation=self.equation,
        )

    @property
    def name(self) -> str:
        return "KolmogorovFlow"

    @property
    def default_params(self) -> KolmogorovFlowParams:
        obs_dim = 2 if self.observation_mode == SIGNED_FORCED_MODE_OBSERVATION_MODE else self.flow.obs_size**2
        kwargs = dict(
            action_dim=4,
            obs_dim=obs_dim,
            k1=self.flow.k,
            k2=self.flow.k,
        )
        for name in (
            "dt",
            "action_time",
            "save_time",
            "reward_alpha",
            "initial_perturbation_amplitude",
            "max_episode_steps",
        ):
            if name in self.env_config:
                kwargs[name] = self.env_config[name]
        return KolmogorovFlowParams(**kwargs)

    def action_space(self, params: Optional[KolmogorovFlowParams] = None):
        params = params or self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(params.action_dim,),
        )

    def observation_space(self, params: KolmogorovFlowParams):
        return spaces.Box(
            low=params.min_obs,
            high=params.max_obs,
            shape=(params.obs_dim,),
        )

    def action_basis_metadata(self, params: Optional[KolmogorovFlowParams] = None) -> Dict[str, object]:
        """Return the versioned velocity-forcing basis for each action index."""
        params = params or self.default_params
        return {
            "version": FORCED_MODE_QUADRATURE_BASIS_VERSION,
            "channels": (
                {"index": 0, "component": "x", "function": "sin", "wavenumber": int(params.k1)},
                {"index": 1, "component": "x", "function": "cos", "wavenumber": int(params.k2)},
                {"index": 2, "component": "x", "function": "sin", "wavenumber": int(params.k3)},
                {"index": 3, "component": "x", "function": "sin", "wavenumber": int(params.k4)},
            ),
        }

    def observation_metadata(self) -> Dict[str, object]:
        """Return the exact observation contract used by :meth:`get_obs`."""
        metadata: Dict[str, object] = {
            "mode": self.observation_mode,
            "version": OBSERVATION_CONTRACT_VERSIONS[self.observation_mode],
        }
        if self.observation_mode == SIGNED_FORCED_MODE_OBSERVATION_MODE:
            metadata.update(
                {
                    "components": ("streamwise_velocity_sine", "streamwise_velocity_cosine"),
                    "wavenumber": int(self.flow.k),
                    "normalization": "two_times_periodic_domain_mean",
                    "trajectory_reduction": "arithmetic_mean",
                }
            )
        return metadata

    def _control_field(self, action: jnp.ndarray, params: KolmogorovFlowParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a1, a2, a3, a4 = action
        forcing_x = (
            a1 * jnp.sin(params.k1 * self.y)
            + a2 * jnp.cos(params.k2 * self.y)
            + a3 * jnp.sin(params.k3 * self.y)
            + a4 * jnp.sin(params.k4 * self.y)
        )
        forcing_y = jnp.zeros_like(self.y)
        return forcing_x, forcing_y

    def _rollout(
        self,
        omega_hat0: jnp.ndarray,
        params: KolmogorovFlowParams,
        control_field: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns:
            final_state_hat, trajectory
        """
        dt = float(params.dt)
        save_n = float(params.save_time)
        action_time = float(params.action_time)

        final_state, trajectory = self.integrator.solve(
            dt=dt,
            flow=self.flow,
            t_span=(0.0, action_time),
            save_n=save_n,
            initial_state=omega_hat0,
            control_field=control_field,
        )
        return final_state, trajectory

    def _calculate_velocity_point(self, omega_hat: jnp.ndarray, i: int, j: int):
        uhat, vhat = compute_velocity_fft(omega_hat, self.kx, self.ky)
        return compute_real_velocity_point(uhat, vhat, i, j)

    def _trajectory_mean_obs(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        stride_x = max(1, self.n // self.flow.obs_size)
        stride_y = max(1, self.m // self.flow.obs_size)

        def obs_one_state(omega_hat):
            # 1. Compute velocity in Fourier space for the whole grid ONCE
            uhat, vhat = compute_velocity_fft(omega_hat, self.kx, self.ky)

            # 2. Inverse FFT for the whole grid ONCE
            ureal = jnp.fft.irfftn(uhat)
            vreal = jnp.fft.irfftn(vhat)

            # 3. Slice exactly at the stride indices to match the 64 grid points
            u_sampled = ureal[::stride_x, ::stride_y]
            v_sampled = vreal[::stride_x, ::stride_y]

            # 4. Compute magnitude and flatten to 1D array
            obs_matrix = jnp.sqrt(jnp.abs(u_sampled) ** 2 + jnp.abs(v_sampled) ** 2)
            return obs_matrix.flatten()

        return jnp.mean(jax.vmap(obs_one_state)(trajectory), axis=0)

    def _trajectory_signed_forced_mode_obs(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        """Project streamwise velocity onto the signed forced-mode quadrature pair.

        The factor of two makes the coefficients amplitude preserving on the
        periodic grid: ``u = A sin(k y) + B cos(k y)`` maps to ``[A, B]``.
        """
        sine_basis = jnp.sin(self.flow.k * self.y)
        cosine_basis = jnp.cos(self.flow.k * self.y)

        def obs_one_state(omega_hat):
            uhat, _ = compute_velocity_fft(omega_hat, self.kx, self.ky)
            streamwise_velocity = jnp.fft.irfftn(uhat, s=self.flow.grid_size)
            return jnp.stack(
                (
                    2.0 * jnp.mean(streamwise_velocity * sine_basis),
                    2.0 * jnp.mean(streamwise_velocity * cosine_basis),
                )
            )

        return jnp.mean(jax.vmap(obs_one_state)(trajectory), axis=0)

    def get_obs(
        self,
        state: KolmogorovFlowState,
        params: KolmogorovFlowParams,
        key: Optional[chex.PRNGKey] = None,
    ) -> chex.Array:
        if self.observation_mode == SIGNED_FORCED_MODE_OBSERVATION_MODE:
            return self._trajectory_signed_forced_mode_obs(state.trajectory)
        return self._trajectory_mean_obs(state.trajectory)

    def _avg_tke(self, trajectory: jnp.ndarray) -> jnp.ndarray:
        def one(omega_hat):
            return compute_tke(omega_hat, self.kx, self.ky, self.n)

        return jnp.mean(jax.vmap(one)(trajectory))

    def _reward(
        self,
        action: jnp.ndarray,
        trajectory: jnp.ndarray,
        params: KolmogorovFlowParams,
    ) -> jnp.ndarray:
        terms = self._reward_terms(action, trajectory, params)
        return terms["tke"] + terms["action_l1"]

    def _reward_terms(
        self,
        action: jnp.ndarray,
        trajectory: jnp.ndarray,
        params: KolmogorovFlowParams,
    ) -> Dict[str, jnp.ndarray]:
        energy = self._avg_tke(trajectory)
        return {
            "tke": -(params.reward_alpha * energy),
            "action_l1": -jnp.sum(jnp.abs(action)),
        }

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: KolmogorovFlowParams,
    ):
        omega0 = self.flow.initial_vorticity(
            key,
            perturbation_amplitude=params.initial_perturbation_amplitude,
        )

        final_state, trajectory = self._rollout(
            omega_hat0=omega0,
            params=params,
            control_field=None,
        )

        state = KolmogorovFlowState(
            trajectory=trajectory,
            omega_hat=final_state,
            time=jnp.array(0),
            terminal=jnp.array(False),
            last_action=jnp.zeros((params.action_dim,), dtype=jnp.real(final_state).dtype),
        )
        obs = self.get_obs(state, params, key)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: KolmogorovFlowState,
        action: chex.Array,
        params: KolmogorovFlowParams,
    ):
        action = self._clip_action(action, params)
        control_field = self._control_field(action, params)

        final_state, trajectory = self._rollout(
            omega_hat0=state.omega_hat,
            params=params,
            control_field=control_field,
        )

        next_state = KolmogorovFlowState(
            trajectory=trajectory,
            omega_hat=final_state,
            time=state.time + 1,
            terminal=jnp.array(False),
            last_action=action,
        )

        obs = self.get_obs(next_state, params, key)
        reward_terms = self._reward_terms(action, trajectory, params)
        reward = reward_terms["tke"] + reward_terms["action_l1"]
        done = self.is_terminal(next_state, params)

        info = {
            "discount": self.discount(next_state, params),
            "mean_tke": self._avg_tke(trajectory),
            "control_l1": jnp.sum(jnp.abs(action)),
            "control_l2": jnp.sum(jnp.square(action)),
            "action_delta_l2": jnp.sum(jnp.square(action - state.last_action)),
            "reward_tke": reward_terms["tke"],
            "reward_action_l1": reward_terms["action_l1"],
            "reward_total": reward,
        }
        return obs, next_state, reward, done, info

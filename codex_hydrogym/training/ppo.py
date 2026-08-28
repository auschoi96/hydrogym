"""Pure-JAX PPO learner for the codex_hydrogym Kolmogorov experiment."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from codex_hydrogym.config import FORCED_MODE_QUADRATURE_BASIS_VERSION, KolmogorovPPOConfig
from codex_hydrogym.modeling.network import ActorCritic
from codex_hydrogym.training.rewards import DeterministicRewardWrapper, compiled_reward_from_config
from hydrogym.jax.env_core import ClipAction, LogWrapper, NormalizeVecObservation, NormalizeVecReward, VecEnv
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict[str, jnp.ndarray]


class RunnerState(NamedTuple):
    """Complete restartable state for one PPO training run."""

    train_state: TrainState
    env_state: Any
    last_observation: jnp.ndarray
    rng: jnp.ndarray
    completed_updates: jnp.ndarray


def build_environment(config: KolmogorovPPOConfig, *, vectorized: bool = True):
    """Create the configured environment and wrappers without hidden defaults."""
    env = KolmogorovFlow(
        env_config=config.environment_config(),
        flow_config=config.flow_config(),
    )
    params = env.default_params
    basis_version = env.action_basis_metadata(params)["version"]
    if basis_version != config.actuation_basis_version or basis_version != FORCED_MODE_QUADRATURE_BASIS_VERSION:
        raise ValueError("configured actuation basis does not match the Kolmogorov environment")
    compiled_reward = compiled_reward_from_config(config)
    if compiled_reward is not None:
        env = DeterministicRewardWrapper(env, compiled_reward)
    env = LogWrapper(env)
    env = ClipAction(env, low=float(params.min_action), high=float(params.max_action))

    if vectorized:
        env = VecEnv(env)
        if config.normalize_environment:
            env = NormalizeVecObservation(env)
            env = NormalizeVecReward(env, config.gamma)
    elif config.normalize_environment:
        raise ValueError("normalize_environment requires vectorized=True")

    return env, params


def _validate_runtime_precision(config: KolmogorovPPOConfig) -> None:
    if config.precision == "float64" and not jax.config.x64_enabled:
        raise RuntimeError("precision='float64' requires JAX_ENABLE_X64=1 before the process starts")


def _build_training_components(config: KolmogorovPPOConfig):
    _validate_runtime_precision(config)
    env, env_params = build_environment(config, vectorized=True)
    network = ActorCritic(env.action_space(env_params).shape[0], activation=config.activation)

    def linear_schedule(count):
        updates_completed = count // (config.num_minibatches * config.update_epochs)
        fraction_remaining = 1.0 - updates_completed / config.num_updates
        return config.learning_rate * fraction_remaining

    learning_rate = linear_schedule if config.anneal_learning_rate else config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1.0e-5),
    )
    return env, env_params, network, optimizer


def _make_initialize(config: KolmogorovPPOConfig, components):
    env, env_params, network, optimizer = components

    def initialize(rng):
        rng, network_rng = jax.random.split(rng)
        init_observation = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(network_rng, init_observation)
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=optimizer)

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, config.num_envs)
        observations, env_state = env.reset(reset_keys, env_params)

        rng, training_rng = jax.random.split(rng)
        return RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_observation=observations,
            rng=training_rng,
            completed_updates=jnp.asarray(0, dtype=jnp.int32),
        )

    return initialize


def make_initialize(config: KolmogorovPPOConfig):
    """Build a pure initializer suitable for ``jax.jit`` and checkpoint templates."""
    return _make_initialize(config, _build_training_components(config))


def _make_update(config: KolmogorovPPOConfig, update_count: int, components):
    if isinstance(update_count, bool) or not isinstance(update_count, int) or update_count <= 0:
        raise ValueError("update_count must be a positive integer")

    env, env_params, network, _ = components

    def update(runner_state):

        def update_step(runner_state, unused):
            def environment_step(step_state, unused):
                current_train_state, current_env_state, last_observation, step_rng = step_state
                step_rng, action_rng = jax.random.split(step_rng)
                policy, value = network.apply(current_train_state.params, last_observation)
                action = policy.sample(seed=action_rng)
                log_probability = policy.log_prob(action)

                step_rng, environment_rng = jax.random.split(step_rng)
                environment_keys = jax.random.split(environment_rng, config.num_envs)
                observation, next_env_state, reward, done, info = env.step(
                    environment_keys,
                    current_env_state,
                    action,
                    env_params,
                )
                transition = Transition(done, action, value, reward, log_probability, last_observation, info)
                next_step_state = (current_train_state, next_env_state, observation, step_rng)
                return next_step_state, transition

            step_state = (
                runner_state.train_state,
                runner_state.env_state,
                runner_state.last_observation,
                runner_state.rng,
            )
            step_state, trajectory = jax.lax.scan(
                environment_step,
                step_state,
                None,
                config.num_steps,
            )

            current_train_state, current_env_state, last_observation, update_rng = step_state
            _, last_value = network.apply(current_train_state.params, last_observation)

            def calculate_gae(trajectory, bootstrap_value):
                def advantage_step(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = transition.reward + config.gamma * next_value * (1 - transition.done) - transition.value
                    gae = delta + config.gamma * config.gae_lambda * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    advantage_step,
                    (jnp.zeros_like(bootstrap_value), bootstrap_value),
                    trajectory,
                    reverse=True,
                    unroll=min(16, config.num_steps),
                )
                return advantages, advantages + trajectory.value

            advantages, targets = calculate_gae(trajectory, last_value)

            def update_epoch(epoch_state, unused):
                def update_minibatch(minibatch_train_state, batch):
                    minibatch_trajectory, minibatch_advantages, minibatch_targets = batch

                    def loss_fn(parameters):
                        policy, value = network.apply(parameters, minibatch_trajectory.obs)
                        log_probability = policy.log_prob(minibatch_trajectory.action)

                        clipped_value = minibatch_trajectory.value + (value - minibatch_trajectory.value).clip(
                            -config.clip_epsilon,
                            config.clip_epsilon,
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - minibatch_targets),
                                jnp.square(clipped_value - minibatch_targets),
                            ).mean()
                        )

                        normalized_advantage = (minibatch_advantages - minibatch_advantages.mean()) / (
                            minibatch_advantages.std() + 1.0e-8
                        )
                        ratio = jnp.exp(log_probability - minibatch_trajectory.log_prob)
                        unclipped_actor = ratio * normalized_advantage
                        clipped_actor = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_epsilon,
                                1.0 + config.clip_epsilon,
                            )
                            * normalized_advantage
                        )
                        actor_loss = -jnp.minimum(unclipped_actor, clipped_actor).mean()
                        entropy = policy.entropy().mean()
                        total_loss = (
                            actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                        )
                        return total_loss, (value_loss, actor_loss, entropy)

                    (total_loss, (value_loss, actor_loss, entropy)), gradients = jax.value_and_grad(
                        loss_fn,
                        has_aux=True,
                    )(minibatch_train_state.params)
                    minibatch_train_state = minibatch_train_state.apply_gradients(grads=gradients)
                    return minibatch_train_state, {
                        "loss_total": total_loss,
                        "loss_value": value_loss,
                        "loss_actor": actor_loss,
                        "entropy": entropy,
                    }

                epoch_train_state, epoch_trajectory, epoch_advantages, epoch_targets, epoch_rng = epoch_state
                epoch_rng, permutation_rng = jax.random.split(epoch_rng)
                permutation = jax.random.permutation(permutation_rng, config.total_batch_size)
                batch = (epoch_trajectory, epoch_advantages, epoch_targets)
                batch = jax.tree_util.tree_map(
                    lambda value: value.reshape((config.total_batch_size,) + value.shape[2:]),
                    batch,
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda value: jnp.take(value, permutation, axis=0),
                    batch,
                )
                minibatches = jax.tree_util.tree_map(
                    lambda value: value.reshape((config.num_minibatches, config.minibatch_size) + value.shape[1:]),
                    shuffled_batch,
                )
                epoch_train_state, loss_metrics = jax.lax.scan(
                    update_minibatch,
                    epoch_train_state,
                    minibatches,
                )
                next_epoch_state = (
                    epoch_train_state,
                    epoch_trajectory,
                    epoch_advantages,
                    epoch_targets,
                    epoch_rng,
                )
                return next_epoch_state, loss_metrics

            epoch_state = (
                current_train_state,
                trajectory,
                advantages,
                targets,
                update_rng,
            )
            epoch_state, loss_metrics = jax.lax.scan(
                update_epoch,
                epoch_state,
                None,
                config.update_epochs,
            )
            current_train_state = epoch_state[0]
            update_rng = epoch_state[-1]

            metrics: dict[str, Any] = dict(trajectory.info)
            metrics.update(loss_metrics)
            next_runner_state = RunnerState(
                train_state=current_train_state,
                env_state=current_env_state,
                last_observation=last_observation,
                rng=update_rng,
                completed_updates=runner_state.completed_updates + 1,
            )
            return next_runner_state, metrics

        runner_state, metrics = jax.lax.scan(
            update_step,
            runner_state,
            None,
            update_count,
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return update


def make_update(config: KolmogorovPPOConfig, update_count: int = 1):
    """Build a pure, restartable PPO update chunk suitable for ``jax.jit``."""
    return _make_update(config, update_count, _build_training_components(config))


def make_train(config: KolmogorovPPOConfig):
    """Build uninterrupted training as initialization plus one full update chunk."""
    components = _build_training_components(config)
    initialize = _make_initialize(config, components)
    update = _make_update(config, config.num_updates, components)

    def train(rng):
        return update(initialize(rng))

    return train

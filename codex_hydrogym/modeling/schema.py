"""Stable observation and action contracts for codex_hydrogym controllers."""

ACTION_DIM = 4
ACTION_MINIMUM = -0.5
ACTION_MAXIMUM = 0.5
ACTION_COLUMNS = tuple(f"action_{index}" for index in range(ACTION_DIM))


def observation_columns(observation_dimension: int) -> tuple[str, ...]:
    if isinstance(observation_dimension, bool) or not isinstance(observation_dimension, int):
        raise TypeError("observation_dimension must be an integer")
    if observation_dimension <= 0:
        raise ValueError("observation_dimension must be positive")
    return tuple(f"observation_{index:04d}" for index in range(observation_dimension))

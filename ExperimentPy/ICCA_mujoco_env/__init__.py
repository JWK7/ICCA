from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

register(
    id="Hand",
    entry_point="gym.envs.mujoco:HandEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

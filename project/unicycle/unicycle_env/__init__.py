from gymnasium.envs.registration import register

register(
    id="unicycle_env/GridWorld-v0",
    entry_point="unicycle_env.envs:GridWorldEnv",
)

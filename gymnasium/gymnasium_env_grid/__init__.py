from gymnasium.envs.registration import register

register(
    id="gymnasium_env_grid/GridWorld-v0",
    entry_point="gymnasium_env_grid.envs:GridWorldEnv",
    kwargs={"render_mode": "human"}
)

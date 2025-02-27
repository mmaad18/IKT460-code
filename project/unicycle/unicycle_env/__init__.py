from gymnasium.envs.registration import register

register(
    id="unicycle_env/UniCycleBasicEnv-v0",
    entry_point="unicycle_env.envs:UniCycleBasicEnv",
    kwargs={"render_mode": "human"}
)

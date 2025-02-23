from gymnasium.envs.registration import register

register(
    id="unicycle_env/CarRacing-v0",
    entry_point="unicycle_env.envs:CarRacing",
    kwargs={"render_mode": "human"}
)

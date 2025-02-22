from gymnasium.envs.registration import register

register(
    id="LidarCoverage-v0",
    entry_point="custom_envs.custom_env:LidarCoverageEnv",
)

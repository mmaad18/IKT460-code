from gymnasium.envs.registration import register

register(
    id="custom_envs/LidarCoverage-v0",
    entry_point="custom_envs.custom_env:LidarCoverageEnv",
)

import gymnasium

import unicycle_env
import numpy as np

from stable_baselines3 import PPO

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
model = PPO.load("ppo_unicycle")

obs, info = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()


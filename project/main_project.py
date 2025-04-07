import gymnasium

import unicycle_env
import numpy as np

from stable_baselines3 import PPO

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
model = PPO.load("ppo_unicycle", device="cpu")

obs, info = env.reset()
for _ in range(20000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()


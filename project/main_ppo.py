import gymnasium
import numpy as np

import unicycle_env

from stable_baselines3 import PPO

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
model = PPO.load("ppo_unicycle", device="cpu")
unwrapped_env = env.unwrapped
env_count = unwrapped_env.get_environment_count()

obs, info = env.reset()
for _ in range(20000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        unwrapped_env.select_environment(np.random.randint(0, env_count))
        obs, info = env.reset()
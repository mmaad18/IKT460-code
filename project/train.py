import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import unicycle_env  # makes sure env is registered


# Create vectorized env (for faster PPO training)
env = make_vec_env("unicycle_env/UniCycleBasicEnv-v0", n_envs=10, seed=42)

# Define and train PPO model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# Train
model.learn(total_timesteps=1_000_000)

# Save model
model.save("ppo_unicycle")

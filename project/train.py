import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import unicycle_env  # makes sure env is registered


# Create vectorized env (for faster PPO training)
vec_env = make_vec_env("unicycle_env/UniCycleBasicEnv-v0", n_envs=10, seed=42)

# Access the actual environments
actual_envs = [env.unwrapped for env in vec_env.envs]

# Define and train PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")

# Train
model.learn(total_timesteps=10_000)

for env in actual_envs:
    env.select_environment(2)

vec_env.reset()

model.learn(total_timesteps=10_000)

# Save model
model.save("ppo_unicycle")

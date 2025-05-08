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

def select_environment(env_id):
    for env in actual_envs:
        env.select_environment(env_id)

    vec_env.reset()

# Train
model.learn(total_timesteps=200_000)
select_environment(2)
model.learn(total_timesteps=200_000)
select_environment(3)
model.learn(total_timesteps=200_000)
select_environment(4)
model.learn(total_timesteps=200_000)
select_environment(5)
model.learn(total_timesteps=200_000)
select_environment(6)
model.learn(total_timesteps=200_000)
select_environment(7)
model.learn(total_timesteps=200_000)
select_environment(8)
model.learn(total_timesteps=200_000)
select_environment(9)
model.learn(total_timesteps=200_000)
select_environment(10)
model.learn(total_timesteps=200_000)
select_environment(11)
model.learn(total_timesteps=200_000)
select_environment(12)
model.learn(total_timesteps=200_000)
select_environment(13)
model.learn(total_timesteps=200_000)


# Save model
model.save("ppo_unicycle")

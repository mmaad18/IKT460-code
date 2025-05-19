import gymnasium
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.monitor import Monitor

import unicycle_env  # makes sure env is registered

# Create log directory
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create vectorized env (for faster PPO training)
vec_env = make_vec_env("unicycle_env/UniCycleBasicEnv-v0", n_envs=10, seed=42, monitor_dir=log_dir)

# Create evaluation environment
eval_env = make_vec_env("unicycle_env/UniCycleBasicEnv-v0", n_envs=1, seed=43)

# Access the actual environments
actual_envs = [env.unwrapped for env in vec_env.envs]

# Define callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=log_dir,
    name_prefix='ppo_unicycle_model'
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Define and train PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", tensorboard_log=log_dir)

def select_environment(env_id):
    for env in actual_envs:
        env.select_environment(env_id)

    vec_env.reset()

# Train with callbacks to track performance
model.learn(total_timesteps=300_000, callback=[checkpoint_callback, eval_callback])

# Save model
model.save("ppo_unicycle")

# Plot training results
plot_results([log_dir], 300_000, "timesteps", "PPO UniCycle Training")
plt.savefig(f"{log_dir}/training_curve.png")
plt.show()

print(f"Training completed. Results saved to {log_dir}")
print(f"You can visualize detailed metrics using TensorBoard:")
print(f"tensorboard --logdir={log_dir}")

import gymnasium
import numpy as np
import torch
import unicycle_env

from project.rl_algorithms.DQN import DQN
from unicycle_env.wrappers import DiscreteActions

action_mapping = [
    [250.0, 0.0],  # Forward
    [-50.0, 0.0],  # Backward
    [0.0, 5.0],    # Turn right
    [0.0, -5.0],   # Turn left
    [250.0, 5.0],  # Forward right
    [250.0, -5.0], # Forward left
    [-50.0, 5.0],  # Backward right
    [-50.0, -5.0], # Backward left
]

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
env = DiscreteActions(env, action_mapping)
unwrapped_env = env.unwrapped
env_count = unwrapped_env.get_environment_count()

state, _ = env.reset()
n_observations = len(state)
n_actions = len(action_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

model = DQN(n_observations, n_actions).to(device)

model.load_state_dict(torch.load('dqn_checkpoint.pth', map_location=device))
model.eval()

obs, info = env.reset()
for _ in range(20000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        action = model(obs_tensor).max(1).indices.item()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        unwrapped_env.select_environment(np.random.randint(0, env_count))
        obs, info = env.reset()
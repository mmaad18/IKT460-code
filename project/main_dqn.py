import gymnasium
import numpy as np
import torch
import unicycle_env

from project.rl_algorithms.DQN import DQN, action_mapping
from unicycle_env.wrappers import DiscreteActions

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

model.load_state_dict(torch.load('dqn_checkpoint_1.pth', map_location=device))
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
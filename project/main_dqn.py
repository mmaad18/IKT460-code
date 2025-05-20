import gymnasium as gym
import numpy as np
import torch
import unicycle_env

from project.rl_algorithms.DQN import DQN, action_mapping
from unicycle_env.wrappers import DiscreteActions

cont_env = gym.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
env = DiscreteActions(cont_env, action_mapping)
unwrapped_env = env.unwrapped
env_count = unwrapped_env.get_environment_count()

state_0, info = env.reset()
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

dqn_agent = DQN(
    device=device,
    input_dim=len(state_0),
    output_dim=len(action_mapping),
    learning_rate=1e-4,
    memory_capacity=10000,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=1000,
    batch_size=128,
    gamma=0.99
)

dqn_agent.policy_net.load_state_dict(torch.load('dqn_checkpoint.pth', map_location=device))
dqn_agent.policy_net.eval()

obs, info = env.reset()
for _ in range(20000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        action = dqn_agent.policy_net(obs_tensor).max(1).indices.item()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        unwrapped_env.select_environment(np.random.randint(0, env_count))
        obs, info = env.reset()


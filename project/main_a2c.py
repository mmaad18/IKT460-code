import gymnasium as gym
import numpy as np
import torch
import unicycle_env
from project.rl_algorithms.A2C import A2C

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

a2c_agent = A2C(123, 8, device, 0.1, 0.1, 3)

a2c_agent.actor.load_state_dict(torch.load('a2c_actor.pth', map_location=device))
a2c_agent.critic.load_state_dict(torch.load('a2c_critic.pth', map_location=device))
a2c_agent.actor.eval()
a2c_agent.critic.eval()

obs, info = env.reset()

#unwrapped_env.select_environment(0)

for _ in range(20000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        action, _, _, _ = a2c_agent.select_action(obs_tensor)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        unwrapped_env.select_environment(np.random.randint(0, env_count))
        obs, info = env.reset()


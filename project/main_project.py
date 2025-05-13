import gymnasium
import numpy as np
import torch

import unicycle_env
# Import your custom DQNAgent
from project.rl_algorithms.DQNAgent import DQNAgent

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
unwrapped_env = env.unwrapped
env_count = unwrapped_env.get_environment_count()

# Get state size from environment
state_size = env.observation_space.shape[0]

# Initialize action handling (same as in training)
if isinstance(env.action_space, gymnasium.spaces.Box):
    # Determine the action space dimensions
    action_dims = env.action_space.shape[0]
    
    # Create discretized actions (5 discrete values per dimension)
    num_discrete_actions_per_dim = 5
    
    # Create discrete actions list
    discrete_actions = []
    action_size = 1  # Start with 1 for multiplication
    
    for dim in range(action_dims):
        low = env.action_space.low[dim]
        high = env.action_space.high[dim]
        
        # Create discrete steps for this dimension
        step_values = np.linspace(low, high, num_discrete_actions_per_dim)
        
        # If this is the first dimension, initialize discrete_actions
        if dim == 0:
            discrete_actions = [[val] for val in step_values]
            action_size = num_discrete_actions_per_dim
        else:
            # For additional dimensions, create combinations
            new_discrete_actions = []
            for existing_action in discrete_actions:
                for val in step_values:
                    new_action = existing_action.copy()
                    new_action.append(val)
                    new_discrete_actions.append(new_action)
            discrete_actions = new_discrete_actions
            action_size *= num_discrete_actions_per_dim
    
    # Convert to numpy arrays
    discrete_actions = [np.array(action) for action in discrete_actions]
else:
    # For discrete action spaces
    action_size = env.action_space.n
    discrete_actions = None

# Initialize DQN agent
agent = DQNAgent(state_size, action_size)

# Load the saved model
agent.policy_net.load_state_dict(torch.load("dqn_checkpoint.pth", map_location=torch.device('cpu')))
# Set the policy network to evaluation mode
agent.policy_net.eval()

obs, info = env.reset()
for _ in range(20000):
    # Get action from DQN agent
    action_idx = agent.select_action(obs)  # Set evaluate=True to disable exploration
    
    # Convert to actual action value if using discretized continuous actions
    if discrete_actions is not None:
        action = discrete_actions[action_idx]
    else:
        action = action_idx
        
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        unwrapped_env.select_environment(np.random.randint(0, env_count))
        obs, info = env.reset()
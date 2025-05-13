import numpy as np
import torch

import gymnasium

import unicycle_env
from project.rl_algorithms.DQNAgent import DQNAgent


def train_dqn(env, agent, n_episodes=1000, max_t=1000, discrete_actions=None):
    """Train DQN agent"""
    scores = []
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # Get discrete action index from agent
            action_idx = agent.select_action(state)
            
            # Convert to actual action value if using discretized continuous actions
            if discrete_actions is not None:
                action = discrete_actions[action_idx]
            else:
                action = action_idx
                
            next_state, reward, done, _, info = env.step(action)
            
            agent.step(state, action_idx, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
                
        scores.append(score)
        
        print(f'Episode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}')
        
        # Check if environment is solved
        if i_episode >= 1000 and np.mean(scores[-100:]) >= 195.0:
            print(f'\nEnvironment solved in {i_episode} episodes!')
            torch.save(agent.policy_net.state_dict(), 'dqn_checkpoint.pth')
            break
            
    return scores


def main():
    env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="rgb_array")
    unwrapped_env = env.unwrapped
    env_count = unwrapped_env.get_environment_count()

    state_size = env.observation_space.shape[0]
    
    # Handle continuous action space by discretizing it
    if isinstance(env.action_space, gymnasium.spaces.Box):
        # Determine the action space dimensions
        action_dims = env.action_space.shape[0]
        
        # Create discretized actions (5 discrete values per dimension)
        # Adjust the number of discrete values based on your needs
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

    # Initialize agent
    agent = DQNAgent(state_size, action_size)

    # Train agent
    scores = train_dqn(env, agent, discrete_actions=discrete_actions)

    env.close()
    print(scores)


if __name__ == "__main__":
    main()
from datetime import datetime
import time
import uuid
from itertools import count

import numpy as np
from tqdm import tqdm

import unicycle_env

import gymnasium as gym

import torch

from project.rl_algorithms.DQN import DQN, action_mapping
from project.rl_algorithms.ReplayMemory import Transition
from project.utils import plot_statistics, coverage_stagnated, save_metadata_json, save_episode_data, logs_path, save_commentary
from unicycle_env.wrappers import DiscreteActions  # pyright: ignore [reportMissingTypeStubs]


def main() -> None:
    cont_env = gym.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="rgb_array")
    env = DiscreteActions(cont_env, action_mapping)
    unwrapped_env = env.unwrapped
    env_count = unwrapped_env.get_environment_count()
    unwrapped_env.select_environment(9)

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
    
    dqn_metadata = dqn_agent.get_metadata()
    run_id = "run_" + str(uuid.uuid4())
    save_metadata_json(dqn_metadata, run_id)
    
    run_time = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d, %H:%M:%S")
    
    save_commentary(f"""
# Comments

### Time of run
{run_time}

### Reward function
self.reward_coefficients = np.array([
            -0.005 / self.dt,  # time
            -0.25 / self.omega_max,  # omega
            -1000.0,  # collision
            1.0 / self.v_max,  # velocity
            50.0,  # coverage
        ], dtype=np.float32)

features = np.array([
            1.0,  # time
            abs(omega),  # omega
            1.0 if _check_collision() else 0.0,  # collision
            v,  # velocity
            delta,  # coverage
        ], dtype=np.float32)
        
R = np.dot(reward_coefficients, features)
    """, run_id)

    TAU = 0.005
    episode_durations = []
    episode_rewards = []
    step_count = 0
    num_episodes = 10000
    episode_max_length = 5000

    for i_episode in tqdm(range(num_episodes)):
        #unwrapped_env.select_environment(np.random.randint(0, env_count))
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        coverage_history = []
        step_infos = []

        for t in count():
            action = dqn_agent.select_action(env, state, step_count)
            step_count += 1
            observation, reward, terminated, truncated, info = env.step(action.item())
            
            step_infos.append(info)
            
            coverage = info['coverage']
            coverage_history.append(coverage)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated or t >= episode_max_length or coverage_stagnated(coverage_history, 3)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            dqn_agent.memory_push(Transition(state, action, next_state, reward))
            state = next_state
            dqn_agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 − τ) θ′
            target_net_state_dict = dqn_agent.target_net.state_dict()
            policy_net_state_dict = dqn_agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

            dqn_agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(reward.item())
                break

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}, average reward: {np.mean(episode_rewards[-100:]):.2f}, "
                  f"average duration: {np.mean(episode_durations[-100:]):.2f}")

            save_episode_data(step_infos, i_episode, run_id)
            
        if i_episode % 1000 == 0 and i_episode > 0:
            torch.save(dqn_agent.policy_net.state_dict(), logs_path(run_id) / f"dqn_checkpoint_{i_episode}.pth")


    torch.save(dqn_agent.policy_net.state_dict(), logs_path(run_id) / f"dqn_checkpoint.pth")
    env.close()
    print("Training complete.")


main()


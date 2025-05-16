import gymnasium
import numpy as np

import unicycle_env

from project.rl_algorithms.MonteCarloControl import MonteCarloControl


def discretize(obs, bins=10):
    return tuple((np.array(obs) * bins).astype(int))

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
estimator = MonteCarloControl(action_space_size=env.action_space.shape[0])

obs, info = env.reset()
episode = []

for episode_idx in range(1000):
    obs, _ = env.reset()
    episode = []

    for t in range(1000):
        state = discretize(obs)
        action = estimator.select_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        obs = next_obs

        if terminated or truncated:
            break

    estimator.update_from_episode(episode)

    if episode_idx % 10 == 0:
        print(f"Episode {episode_idx}, unique states: {len(estimator.Q)}")



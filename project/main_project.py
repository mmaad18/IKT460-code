import gymnasium

import unicycle_env
import numpy as np

env = gymnasium.make('unicycle_env/CarRacing-v0')

observation, info = env.reset()
for _ in range(1000):
    # this is where you would insert your policy
    action = np.array([-0.05, 0.2, 0.0])

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()


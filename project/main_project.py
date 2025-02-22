import gymnasium as gym

from unicycle_env import LidarRobotEnv

# Initialise the environment
# env = gym.make("CarRacing-v3", render_mode="human")
env = LidarRobotEnv()

# Reset the environment to generate the first observation
observation, info = env.reset()
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()




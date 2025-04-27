import gymnasium
import unicycle_env

from project.rl_algorithms.MonteCarloControl import MonteCarloControl


estimator = MonteCarloControl(gamma=0.99)

env = gymnasium.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="human")
episode = []
obs, info = env.reset()

for step in range(20000):
    action = policy(obs)  # Replace this with random or hand-coded policy
    next_obs, reward, terminated, truncated, info = env.step(action)

    episode.append((obs, reward))
    obs = next_obs

    if terminated or truncated:
        estimator.evaluate(episode)
        print(f"Episode done. Unique states seen: {len(estimator.V)}")
        obs, info = env.reset()
        episode = []

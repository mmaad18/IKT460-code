import matplotlib.pyplot as plt
import numpy as np
from udemy.optimistic_initial_values.Bandit import Bandit


def experiment():
    NUM_TRIALS = 10000
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):

        # Use optimistic initial values to select the next bandit
        j = np.argmax([b.p_hat for b in bandits])

        # Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        # Update rewards history
        rewards[i] = x

        # Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # Print mean estimation for each bandit
    for b in bandits:
        print(f"mean estimate: {b.p_hat}")

    # Print total reward
    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.ylim([0, 1])
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()
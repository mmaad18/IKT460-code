import matplotlib.pyplot as plt
import numpy as np
from udemy.ucb1.Bandit import Bandit


def ucb(mean, N, n_j):
    return mean + np.sqrt(2 * np.log(N) / n_j)


def experiment():
    NUM_TRIALS = 10000
    BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    total_plays = 0

    # Initialization: Play each bandit once
    for bandit in bandits:
        x = bandit.pull()
        bandit.update(x)
        total_plays += 1

    for i in range(NUM_TRIALS):
        j = np.argmax([ucb(b.p_hat, total_plays, b.N) for b in bandits])

        x = bandits[j].pull()
        bandits[j].update(x)
        total_plays += 1

        # Update rewards history
        rewards[i] = x


    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

    # plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.show()

    # plot moving average ctr linear
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
    plt.show()

    for b in bandits:
        print(b.p_hat)

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum() / NUM_TRIALS)
    print("num times selected each bandit:", [b.N for b in bandits])

    return cumulative_average


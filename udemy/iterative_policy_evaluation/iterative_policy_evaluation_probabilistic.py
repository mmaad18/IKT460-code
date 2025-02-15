import numpy as np

from udemy.environments.Action import Action
from udemy.environments.GridWorldWindy import GridWorldWindy


def print_policy(policy, grid):
    for i in range(grid.shape[0]):
        print("--------------------------------------------")
        for j in range(grid.shape[1]):
            actions = policy[i, j]
            if actions is None:
                print("  X: 0.0  |", end="")
            else:
                action_strings = [f"{action[0]}: {action[1]:.1f}" for action in actions]
                print("  " + ", ".join(action_strings) + "  |", end="")
        print("")


def print_values(values, grid):
    for i in range(grid.shape[0]):
        print("------------------------------------------------")
        for j in range(grid.shape[1]):
            v = values[i, j]
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")


def get_values(gridWorld, policy, gamma=0.9, delta=1e-3) -> np.ndarray:
    grid = gridWorld.grid
    V = np.zeros(grid.shape)

    biggest_change = float("inf")
    counter = 0

    while biggest_change >= delta:
        biggest_change = 0
        new_V = np.copy(V)

        for i, j in np.ndindex(grid.shape):
            actions = policy[i, j]

            if actions is not None:
                new_value = 0
                gridWorld.position = (i, j)

                for action, transition_prob in actions:
                    for action_prob in gridWorld.action_probabilities[i, j]:
                        next_state = gridWorld.simulate(action)
                        reward = gridWorld.rewards[next_state]

                        if len(gridWorld.action_probabilities[i, j]) > 1:
                            new_value += gridWorld.action_probabilities[i, j][action] * transition_prob * (reward + gamma * V[next_state])
                        else:
                            new_value += transition_prob * (reward + gamma * V[next_state])

                biggest_change = max(biggest_change, abs(new_value - V[i, j]))
                new_V[i, j] = new_value

        V = new_V
        counter += 1
        print(f"Counter: {counter}")

    return V


def experiment():
    gridWorld = GridWorldWindy((3,4), (2,0))
    grid = gridWorld.grid

    policy = np.full(grid.shape, None, dtype=object)
    policy[0, 0] = [(Action.RIGHT, 1.0)]
    policy[0, 1] = [(Action.RIGHT, 1.0)]
    policy[0, 2] = [(Action.RIGHT, 1.0)]
    policy[1, 0] = [(Action.UP, 1.0)]
    policy[1, 2] = [(Action.UP, 1.0)]
    policy[2, 0] = [(Action.UP, 0.5), (Action.RIGHT, 0.5)]
    policy[2, 1] = [(Action.RIGHT, 1.0)]
    policy[2, 2] = [(Action.UP, 1.0)]
    policy[2, 3] = [(Action.LEFT, 1.0)]

    print_policy(policy, grid)

    values = get_values(gridWorld, policy)
    print_values(values, grid)

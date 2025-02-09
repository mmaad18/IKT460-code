import numpy as np

from udemy.environments.Action import Action
from udemy.environments.GridWorld import GridWorld


def print_policy(policy, grid):
    for i in range(grid.shape[0]):
        print("------------------------")
        for j in range(grid.shape[1]):
            action = policy[i, j]
            if action is None:
                print("  X  |", end="")
            else:
                print("  %s  |" % action, end="")
        print("")


def print_values(values, grid):
    for i in range(grid.shape[0]):
        print("---------------------------")
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
            action = policy[i, j]

            if action is not None:
                gridWorld.state = (i, j)
                next_state = gridWorld.step(action)

                reward = gridWorld.get_reward()
                new_value = reward + gamma * V[next_state]

                biggest_change = max(biggest_change, abs(new_value - V[i, j]))

                new_V[i, j] = new_value

        V = new_V
        counter += 1
        print(f"Counter: {counter}")

    return V


def experiment():
    gridWorld = GridWorld((3,4), (2,0))
    grid = gridWorld.grid

    policy = np.full(grid.shape, None, dtype=object)
    policy[0, 0] = Action.RIGHT
    policy[0, 1] = Action.RIGHT
    policy[0, 2] = Action.RIGHT
    policy[1, 0] = Action.UP
    policy[1, 2] = Action.UP
    policy[2, 0] = Action.UP
    policy[2, 1] = Action.RIGHT
    policy[2, 2] = Action.UP
    policy[2, 3] = Action.LEFT

    print_policy(policy, grid)

    values = get_values(gridWorld, policy)
    print_values(values, grid)


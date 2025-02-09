import numpy as np

from udemy.environments.Action import Action
from udemy.environments.GridWorld import GridWorld


def print_policy(policy, grid):
    for i in range(grid.shape[0]):
        print("------------------------")
        for j in range(grid.shape[1]):
            a = policy[i, j]

            if a is None:
                print("  X  |", end="")
            else:
                print("  %s  |" % a, end="")
        print("")


def print_values(values, grid):
    for i in range(grid.shape[0]):
        print("---------------------------")
        for j in range(grid.shape[1]):
            v = values.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")


def get_values(grid, action, gamma=0.9) -> np.ndarray:
    V = np.zeros(grid.shape)

    delta: float = 1e-3

    return V


def experiment():
    THRESHOLD = 1e-3

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


    a = Action.UP
    #b = a.reverse()

    print_policy(policy, grid)
    V = get_values(grid, policy)


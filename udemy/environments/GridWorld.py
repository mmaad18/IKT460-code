import numpy as np
from udemy.environments.Action import Action
from udemy.environments.CellType import CellType
from udemy.environments.GridCell import GridCell


class GridWorld:
    def __init__(self, shape: tuple[int, int], start: tuple[int, int]):
        self.shape = shape
        self.position = start
        self.grid = np.ones(shape, dtype=object)
        self.rewards = 0

        for i, j in np.ndindex(shape):
            self.grid[i, j] = GridCell(CellType.PATH, np.zeros(4), np.eye(4))

        self.grid[start] = GridCell(CellType.START, np.zeros(4), np.eye(4))


    def print(self):
        border = "-" * (4 * self.shape[1] + 1)
        print(border)

        for i in range(self.shape[0]):
            row = " | ".join(str(self.grid[i, j].cell_type) for j in range(self.shape[1]))
            print(f"| {row} |")
            print(border)


    def print_policy(self, policy):
        border = "-" * (4 * self.shape[1] + 1)

        for i in range(self.shape[0]):
            print(border + "\n|", end="")
            for j in range(self.shape[1]):
                action = policy[i, j]
                if action is None:
                    print(" %s |" % self.grid[i, j].cell_type, end="")
                else:
                    print(" %s |" % action, end="")
            print("")
        print(border)


    def get_values(self, policy, gamma=0.9, delta=1e-3) -> np.ndarray:
        grid = self.grid
        V = np.zeros(grid.shape)

        biggest_change = float("inf")
        counter = 0

        while biggest_change >= delta:
            biggest_change = 0
            new_V = np.copy(V)

            for i, j in np.ndindex(grid.shape):
                action = policy[i, j]

                if action is not None:
                    self.position = (i, j)
                    next_state = self.step(action)

                    reward = self.get_reward()
                    new_value = reward + gamma * V[next_state]

                    biggest_change = max(biggest_change, abs(new_value - V[i, j]))

                    new_V[i, j] = new_value

            V = new_V
            counter += 1
            print(f"Counter: {counter}")

        return V


    def _next_state(self, action: Action) -> (tuple[int, int]):
        x, y = self.position
        dx, dy = action.step()

        x = max(0, min(self.grid.shape[0] - 1, x + dx))
        y = max(0, min(self.grid.shape[1] - 1, y + dy))

        return (x, y)


    def step(self, action: Action) -> (tuple[int, int]):
        x, y = self._next_state(action)

        if self.grid[x, y]:
            self.position = (x, y)

        return self.position


    def simulate(self, action: Action) -> (tuple[int, int]):
        x, y = self._next_state(action)

        if self.grid[x, y]:
            return (x, y)

        return self.position


    def get_reward(self) -> float:
        return self.rewards[self.position]


    def reset(self, start: tuple[int, int]) -> (tuple[int, int]):
        self.position = start
        return self.position


    def is_end(self, end: tuple) -> bool:
        return self.position == end


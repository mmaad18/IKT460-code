import numpy as np
from udemy.environments.Action import Action


class GridWorld:
    def __init__(self, size: tuple, start: tuple):
        self.grid = np.zeros(size, dtype=bool)
        self.reward = np.zeros(size)
        self.state = start

        # Obstacles
        self.grid[1, 1] = True

        # Rewards
        self.reward[:, :] = -0.1
        self.reward[0, 3] = 1
        self.reward[1, 3] = -1


    def step(self, action: Action) -> (tuple):
        x, y = self.state
        dx, dy = action.move

        x = max(0, min(self.grid_size[0] - 1, x + dy))
        y = max(0, min(self.grid_size[1] - 1, y + dx))

        if not self.grid[x, y]:
            self.state = (x, y)

        return self.state


    def reset(self, start: tuple) -> (tuple):
        self.state = start
        return self.state


    def is_end(self, end: tuple) -> bool:
        return self.state == end


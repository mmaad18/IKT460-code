import numpy as np
from udemy.environments.Action import Action


class GridWorldWindy:
    def __init__(self, size: tuple, start: tuple):
        self.grid = np.ones(size, dtype=bool)
        self.rewards = np.zeros(size)
        self.action_probabilities = np.zeros(size, dtype=set)
        self.state = start

        # Obstacles
        self.grid[1, 1] = False

        # Rewards
        self.rewards[:, :] = -0.0
        self.rewards[0, 3] = 1
        self.rewards[1, 3] = -1

        # Action probabilities
        self.action_probabilities[:, :] = {1.0}
        self.action_probabilities[1, 2] = {Action.UP: 0.5, Action.RIGHT: 0.5}


    def step(self, action: Action) -> (tuple):
        x, y = self.state
        dx, dy = action.move()

        x = max(0, min(self.grid.shape[0] - 1, x + dx))
        y = max(0, min(self.grid.shape[1] - 1, y + dy))

        if self.grid[x, y]:
            self.state = (x, y)

        return self.state


    def simulate(self, action: Action) -> (tuple):
        x, y = self.state
        dx, dy = action.move()

        x = max(0, min(self.grid.shape[0] - 1, x + dx))
        y = max(0, min(self.grid.shape[1] - 1, y + dy))

        if self.grid[x, y]:
            return (x, y)

        return self.state


    def get_reward(self) -> float:
        return self.rewards[self.state]


    def reset(self, start: tuple) -> (tuple):
        self.state = start
        return self.state


    def is_end(self, end: tuple) -> bool:
        return self.state == end


import numpy as np
from udemy.environments.Action import Action
from udemy.environments.CellType import CellType
from udemy.environments.BaseCell import BaseCell


class GridCell(BaseCell):
    def __init__(
            self,
            cell_type: CellType,
            out_rewards: np.ndarray, # 4x1
            out_probabilities: np.ndarray # 4x4
    ):
        super().__init__(cell_type)
        # UP, DOWN, LEFT, RIGHT
        self.out_rewards = out_rewards
        self.out_probabilities = out_probabilities

        if not np.allclose(out_probabilities.sum(axis=1), 1.0):
            raise ValueError("Each row in out_probabilities must sum to 1.")

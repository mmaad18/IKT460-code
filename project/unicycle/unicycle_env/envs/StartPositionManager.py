from pathlib import Path

import numpy as np


class StartPositionManager:
    def __init__(self, start_positions_path: Path) -> None:
        self.idx = 0
        self.starting_positions_count = -1
        self.starting_positions = np.load(start_positions_path)
        self.starting_positions_count = len(self.starting_positions)


    def next(self) -> tuple[int, int]:
        self.idx = (self.idx + 1) % self.starting_positions_count
        return tuple(self.starting_positions[self.idx])




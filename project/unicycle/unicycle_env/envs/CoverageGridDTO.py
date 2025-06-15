from dataclasses import dataclass, field

from numpy.typing import NDArray

import numpy as np
from pygame.color import Color


@dataclass
class CoverageGridDTO:
    map_dimensions: tuple[int, int]
    resolution: int
    color: Color = field(init=False)
    grid: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        w, h = self.map_dimensions
        self.color = Color("lightblue")
        self.grid = np.zeros((w // self.resolution, h // self.resolution), dtype=bool)


    def mark_visited(self, position: tuple[float, float]) -> None:
        x, y = position
        grid_x = int(x // self.resolution)
        grid_y = int(y // self.resolution)

        if not self.grid[grid_x, grid_y]:
            self.grid[grid_x, grid_y] = True


    def coverage(self) -> int:
        return int(np.sum(self.grid))


    def coverage_percentage(self) -> float:
        return float(np.sum(self.grid)) / self.grid.size



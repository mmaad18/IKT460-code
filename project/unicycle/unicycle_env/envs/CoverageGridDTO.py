from dataclasses import dataclass, field

import numpy as np
from pygame.color import Color


@dataclass
class CoverageGridDTO:
    map_dimensions: tuple[int, int]
    resolution: int
    color: Color = field(init=False)
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        w, h = self.map_dimensions
        self.color = Color("lightblue")
        self.grid = np.zeros((w // self.resolution, h // self.resolution), dtype=bool)


    def visited(self, position: tuple[float, float]) -> bool:
        x, y = position
        grid_x = int(x // self.resolution)
        grid_y = int(y // self.resolution)

        if not self.grid[grid_x, grid_y]:
            self.grid[grid_x, grid_y] = True

        return self.grid[grid_x, grid_y]


    def coverage(self) -> int:
        return np.sum(self.grid)


    def coverage_percentage(self) -> float:
        return np.sum(self.grid) / self.grid.size



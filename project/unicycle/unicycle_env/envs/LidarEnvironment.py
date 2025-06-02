import numpy as np
import pygame
from pygame.color import Color
from skimage.io import imread  # pyright: ignore [reportMissingTypeStubs, reportUnknownVariableType]
from numpy.typing import NDArray

from unicycle_env.envs.Agent import Agent  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.CoverageGridDTO import CoverageGridDTO  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.ObstacleDTO import ObstacleDTO  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.StartPositionManager import StartPositionManager  # pyright: ignore [reportMissingTypeStubs]


class LidarEnvironment:
    def __init__(self, map_image_path: str, start_positions_path: str, map_dimensions: tuple[int, int], map_window_name: str="LIDAR SIM") -> None:
        pygame.init()
        pygame.display.set_caption(map_window_name)

        self.start_position_manager = StartPositionManager(start_positions_path)

        self.surface_load = pygame.image.load(map_image_path)
        self.map_dimensions = map_dimensions
        self.surface = pygame.display.set_mode(map_dimensions)
        self.surface.blit(self.surface_load, (0, 0))

        self.lidar_surface = self.surface.copy()
        self.dynamic_obstacles: list[ObstacleDTO] = []

        image_array: NDArray[np.uint8] = imread(map_image_path)
        bw_image: NDArray[np.float64] = np.dot(image_array[..., :3], np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]))
        self.walls: NDArray[np.bool_] = (bw_image < 128).T


    def next_start_position(self) -> tuple[int, int]:
        return self.start_position_manager.next()


    def update(self, agent: Agent, coverage_grid: CoverageGridDTO, measurements: NDArray[np.float32]) -> None:
        self.surface.blit(self.surface_load, (0, 0))
        self.draw_coverage_grid(coverage_grid)
        self.move_obstacles()
        self.draw_obstacles()
        self.draw_lidar_data(measurements)
        self.draw_agent(agent)
        pygame.display.update()


    def get_size(self) -> tuple[int, int]:
        return self.map_dimensions


    def get_at(self, position: tuple[int, int]) -> bool:
        return self.walls[position]


    def get_walls(self) -> NDArray[np.bool_]:
        return self.walls


    """
    COVERAGE GRID
    """
    def draw_coverage_grid(self, coverage_grid: CoverageGridDTO) -> None:
        grid = coverage_grid.grid
        res = coverage_grid.resolution
        color = coverage_grid.color

        for x, y in np.argwhere(grid):
            rect = pygame.Rect(x * res, y * res, res, res)
            pygame.draw.rect(self.surface, color, rect)


    """
    AGENT
    """
    def draw_lidar_data(self, measurements: NDArray[np.float32], point_radius: int = 2) -> None:
        self.lidar_surface = self.surface.copy()

        for m in measurements:
            color = Color("red") if m[2] == 1.0 else Color("lightpink")
            position = (int(m[3]), int(m[4]))
            pygame.draw.circle(self.lidar_surface, color, position, int(point_radius))

        self.surface.blit(self.lidar_surface, (0, 0))


    def draw_agent(self, agent: Agent) -> None:
        polygon = agent.get_polygon()
        pygame.draw.polygon(self.surface, agent.color, polygon)


    """
    OBSTACLES
    """
    def add_obstacle(self, position: tuple[int, int], size: tuple[int, int], color: Color = Color(0, 0, 0)) -> None:
        self.dynamic_obstacles.append(ObstacleDTO(position, size, color))


    def move_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            ox, oy = obstacle.position
            obstacle.position = (ox + 2, oy)  # Move right


    def draw_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            pygame.draw.rect(self.surface, obstacle.color, (obstacle.position, obstacle.size))




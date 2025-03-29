import numpy as np
import pygame
from pygame.color import Color

from unicycle_env.envs.CoverageGridDTO import CoverageGridDTO
from unicycle_env.envs.ObstacleDTO import ObstacleDTO
from unicycle_env.envs.MeasurementDTO import MeasurementDTO
from unicycle_env.envs.AgentDTO import AgentDTO


class LidarEnvironment:
    def __init__(self, map_image_path: str, map_dimensions: tuple[int, int], map_window_name="LIDAR SIM"):
        pygame.init()
        pygame.display.set_caption(map_window_name)

        self.surface_load = pygame.image.load(map_image_path)
        self.map_dimensions = map_dimensions
        self.surface = pygame.display.set_mode(map_dimensions)
        self.surface.blit(self.surface_load, (0, 0))

        self.lidar_surface = self.surface.copy()
        self.dynamic_obstacles: list[ObstacleDTO] = []


    def update(self, agent: AgentDTO, coverage_grid: CoverageGridDTO, lidar_data: list[MeasurementDTO]):
        self.surface.blit(self.surface_load, (0, 0))
        self.draw_coverage_grid(coverage_grid)
        self.move_obstacles()
        self.draw_obstacles()
        self.draw_lidar_data(lidar_data)
        self.draw_agent(agent)
        pygame.display.update()


    def get_size(self) -> tuple[int, int]:
        return self.map_dimensions


    def get_at(self, position: tuple[int, int]) -> Color:
        """ Returns pixel color at a given position (checks obstacles first) """
        x, y = position

        for obstacle in self.dynamic_obstacles:
            xo, yo = obstacle.position
            w, h = obstacle.size

            # Inside obstacle
            if xo <= x <= xo + w and yo <= y <= yo + h:
                return obstacle.color

        return self.surface.get_at((int(x), int(y)))


    """
    COVERAGE GRID
    """
    def draw_coverage_grid(self, coverage_grid: CoverageGridDTO):
        grid = coverage_grid.grid
        res = coverage_grid.resolution
        color = coverage_grid.color

        for x, y in np.argwhere(grid):
            rect = pygame.Rect(x * res, y * res, res, res)
            pygame.draw.rect(self.surface, color, rect)


    """
    AGENT
    """
    def draw_lidar_data(self, lidar_data: list[MeasurementDTO], point_radius: int = 2):
        self.lidar_surface = self.surface.copy()

        for m in lidar_data:
            x, y = m.to_cartesian()
            color = Color("red") if m.hit == 1.0 else Color("lightpink")
            pygame.draw.circle(self.lidar_surface, color, (round(x), round(y)), int(point_radius))

        self.surface.blit(self.lidar_surface, (0, 0))


    def draw_agent(self, agent: AgentDTO):
        polygon = agent.get_polygon()
        pygame.draw.polygon(self.surface, agent.color, polygon)


    """
    OBSTACLES
    """
    def add_obstacle(self, position: tuple[int, int], size: tuple[int, int], color=(0, 0, 0)):
        self.dynamic_obstacles.append(ObstacleDTO(position, size, color))


    def move_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            ox, oy = obstacle.position
            obstacle.position = (ox + 2, oy)  # Move right


    def draw_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            pygame.draw.rect(self.surface, obstacle.color, (obstacle.position, obstacle.size))




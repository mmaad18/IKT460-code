import math
import pygame


def lidar_data_2_pos(distance, angle, position):
    x = distance * math.cos(angle) + position[0]
    y = - distance * math.sin(angle) + position[1]

    return int(x), int(y)


class LidarEnvironment:
    def __init__(self, map_image_path: str, map_dimensions: (int, int), map_window_name="LIDAR SIM"):
        pygame.init()
        pygame.display.set_caption(map_window_name)

        self.surface_load = pygame.image.load(map_image_path)
        self.map_dimensions = map_dimensions
        self.surface = pygame.display.set_mode(map_dimensions)
        self.surface.blit(self.surface_load, (0, 0))

        self.lidar_surface = self.surface.copy()

        self.dynamic_obstacles = []

    def update(self, data):
        self.move_obstacles()
        self.draw_obstacles()
        self.draw_lidar_data(data)
        pygame.display.update()

    def get_size(self) -> tuple:
        return self.map_dimensions

    def get_at(self, position):
        """ Returns pixel color at a given position (checks obstacles first) """
        for obstacle in self.dynamic_obstacles:
            x, y = position
            ox, oy = obstacle["pos"]
            w, h = obstacle["size"]

            if ox <= x <= ox + w and oy <= y <= oy + h:  # Inside obstacle
                return obstacle["color"]

        return self.surface.get_at((int(position[0]), int(position[1])))

    def draw_lidar_data(self, data):
        self.lidar_surface = self.surface.copy()

        for element in data:
            point = lidar_data_2_pos(element[0], element[1], element[2])

            self.lidar_surface.set_at((int(point[0]), int(point[1])), (255, 0, 0))

    """
    OBSTACLES
    """
    def add_obstacle(self, position=(0, 0), size=(10, 10), color=(0, 0, 0)):
        self.dynamic_obstacles.append({"pos": position, "size": size, "color": color})

    def move_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            obstacle["pos"] = (obstacle["pos"][0] + 2, obstacle["pos"][1])  # Moves right

    def draw_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            pygame.draw.rect(self.surface, obstacle["color"], (obstacle["pos"], obstacle["size"]))




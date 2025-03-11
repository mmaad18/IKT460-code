import math
import pygame


class LidarEnvironment:
    def __init__(self, map_image_path, MapDimensions):
        pygame.init()

        self.externalmap = pygame.image.load(map_image_path)
        self.mapW, self.mapH = MapDimensions
        self.map = pygame.display.set_mode((self.mapW, self.mapH))
        self.map.blit(self.externalmap, (0, 0))

        self.dynamic_obstacles = []

    def add_obstacle(self, position, size=(10, 10), color=(0, 0, 0)):
        """ Adds a new obstacle to the dynamic obstacles list """
        self.dynamic_obstacles.append({"pos": position, "size": size, "color": color})

    def move_obstacles(self):
        """ Updates obstacle positions (simple movement for now) """
        for obstacle in self.dynamic_obstacles:
            obstacle["pos"] = (obstacle["pos"][0] + 2, obstacle["pos"][1])  # Moves right

    def draw_obstacles(self):
        """ Draws obstacles on the environment """
        for obstacle in self.dynamic_obstacles:
            pygame.draw.rect(self.map, obstacle["color"], (obstacle["pos"], obstacle["size"]))

    def get_size(self):
        """ Returns the map dimensions (W, H). Used by LiDAR """
        return self.mapW, self.mapH

    def get_at(self, position):
        """ Returns pixel color at a given position (checks obstacles first) """
        for obstacle in self.dynamic_obstacles:
            x, y = position
            ox, oy = obstacle["pos"]
            w, h = obstacle["size"]

            if ox <= x <= ox + w and oy <= y <= oy + h:  # Inside obstacle
                return obstacle["color"]

        return self.map.get_at((int(position[0]), int(position[1])))

    def update(self):
        """ Updates and redraws obstacles """
        self.move_obstacles()
        self.draw_obstacles()
        pygame.display.update()

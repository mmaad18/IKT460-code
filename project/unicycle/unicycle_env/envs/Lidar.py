import math

import numpy as np
from pygame.color import Color

from unicycle_env.envs.Agent import Agent
from unicycle_env.envs.LidarEnvironment import LidarEnvironment
from unicycle_env.envs.MeasurementDTO import MeasurementDTO


class Lidar:
    def __init__(self, environment: LidarEnvironment, max_distance: int, num_rays: int, uncertainty: tuple[float, float]):
        self.environment = environment
        self.max_distance = max_distance
        self.num_rays = num_rays
        self.width, self.height = environment.get_size()
        self.sigma = uncertainty


    def measurement(
            self,
            agent: Agent,
            step: int = 2
    ) -> list[MeasurementDTO]:
        measurements = []
        x1, y1 = agent.position

        for relative_angle in np.linspace(0, 2 * np.pi, self.num_rays, False):
            global_angle = (agent.angle + relative_angle) % (2 * np.pi)
            cos_a = math.cos(global_angle)
            sin_a = math.sin(global_angle)
            hit = 0.0

            for distance in range(0, self.max_distance, step):
                x2 = int(x1 + distance * cos_a)
                y2 = int(y1 - distance * sin_a)

                if 0 <= x2 < self.width and 0 <= y2 < self.height:
                    color = self.environment.get_at((x2, y2))
                    if color == Color("black"):
                        hit = 1.0
                        clean_measurement = MeasurementDTO(distance, relative_angle, hit, (x2, y2), agent)
                        noisy_measurement = clean_measurement.with_uncertainty(self.sigma[0], self.sigma[1])
                        measurements.append(noisy_measurement)
                        break

            if hit == 0.0:
                x2 = int(x1 + self.max_distance * cos_a)
                y2 = int(y1 - self.max_distance * sin_a)
                max_measurement = MeasurementDTO(self.max_distance, relative_angle, hit, (x2, y2), agent)
                measurements.append(max_measurement)

        return measurements










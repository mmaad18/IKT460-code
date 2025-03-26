import math
import numpy as np
from unicycle_env.envs.LidarEnvironment import LidarEnvironment
from unicycle_env.envs.MeasurementDTO import MeasurementDTO


class Lidar:
    def __init__(self, environment: LidarEnvironment, max_distance: int, uncertainty: tuple[int, float]):
        self.environment = environment
        self.max_distance = max_distance
        self.width, self.height = environment.get_size()
        self.sigma = np.array(uncertainty, dtype=np.float32)


    def measurement(
            self,
            position: tuple[float, float],
            num_rays: int,
            step: int = 2
    ) -> list[MeasurementDTO]:
        measurements = []
        x1, y1 = position

        for angle in np.linspace(0, 2 * np.pi, num_rays, False):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            for distance in range(0, self.max_distance, step):
                x2 = round(x1 + distance * cos_a)
                y2 = round(y1 - distance * sin_a)

                if 0 <= x2 < self.width and 0 <= y2 < self.height:
                    color = self.environment.get_at((x2, y2))
                    if color[:3] == (0, 0, 0):
                        measurements.append(MeasurementDTO(distance, angle, position))
                        break

        return measurements










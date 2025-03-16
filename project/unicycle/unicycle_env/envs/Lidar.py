import math
import numpy as np
from project.unicycle.unicycle_env.envs.LidarEnvironment import LidarEnvironment


def add_uncertainty(distance, angle, sigma):
    noisy_distance = max(np.random.normal(distance, sigma[0]), 0)
    noisy_angle = angle + np.random.normal(0, sigma[1])

    return [noisy_distance, noisy_angle]


class Lidar:
    def __init__(self, environment: LidarEnvironment, distance: int, uncertainty: (int, int)):
        self.environment = environment
        self.distance = distance
        self.sigma = np.array(uncertainty, dtype=np.float32)
        self.width, self.height = environment.get_size()


    def measurement(self, position, num_rays=60):
        data = []
        x1, y1 = position

        for angle in np.linspace(0, 2 * np.pi, num_rays, False):
            for r in range(0, self.distance, 2):
                x2 = int(x1 + r * math.cos(angle))
                y2 = int(y1 - r * math.sin(angle))

                if 0 <= x2 < self.width and 0 <= y2 < self.height:
                    color = self.environment.get_at((x2, y2))

                    if color[:3] == (0, 0, 0):
                        noisy_measurement = add_uncertainty(r, angle, self.sigma)
                        noisy_measurement.append(position)
                        data.append(noisy_measurement)
                        break

        return data if data else False









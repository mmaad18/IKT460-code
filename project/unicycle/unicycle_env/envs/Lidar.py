import numpy as np
from numpy.typing import NDArray

from cython_module.lidar_core import lidar_measurement  # pyright: ignore [reportUnknownVariableType]
from unicycle_env.envs.Agent import Agent  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.LidarEnvironment import LidarEnvironment  # pyright: ignore [reportMissingTypeStubs]


class Lidar:
    def __init__(self, environment: LidarEnvironment, max_distance: int, num_rays: int, uncertainty: tuple[float, float]) -> None:
        self.environment = environment
        self.max_distance = max_distance
        self.num_rays = num_rays
        self.width, self.height = environment.get_size()
        self.sigma_distance, self.sigma_angle = uncertainty
        self.relative_angles = np.linspace(0, 2 * np.pi, self.num_rays, False)

        # distance, angle, hit, x2, y2
        self.measurements = np.zeros((self.num_rays, 5), dtype=np.float32)


    """
    Measurement: distance, angle, hit, x2, y2
    """
    def measurement(
            self,
            agent: Agent,
            step: int = 2
    ) -> NDArray[np.float32]:
        x, y = agent.position
        angle = agent.angle
        lidar_measurement(self.measurements, self.relative_angles, x, y, angle,
                          self.width, self.height, self.max_distance, step,
                          self.environment.get_walls())

        return self.measurements










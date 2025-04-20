from dataclasses import dataclass

import numpy as np

from unicycle_env.envs.Agent import Agent


@dataclass
class MeasurementDTO:
    distance: float
    angle: float
    hit: float
    position: tuple[int, int]
    agent: Agent

    def with_uncertainty(self, sigma_distance: float, sigma_angle: float) -> "MeasurementDTO":
        noisy_distance = max(0.0, np.random.normal(self.distance, sigma_distance))
        noisy_angle = np.random.normal(self.angle, sigma_angle)

        return MeasurementDTO(noisy_distance, noisy_angle, self.hit, self.position, self.agent)


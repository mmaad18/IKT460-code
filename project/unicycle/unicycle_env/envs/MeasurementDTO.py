from dataclasses import dataclass
import numpy as np

from unicycle_env.envs.AgentDTO import AgentDTO


@dataclass
class MeasurementDTO:
    distance: float
    angle: float
    hit: float
    agent: AgentDTO

    def to_cartesian(self) -> tuple[float, float]:
        global_angle = (self.agent.angle + self.angle) % (2 * np.pi)
        x = self.agent.position[0] + self.distance * np.cos(global_angle)
        y = self.agent.position[1] - self.distance * np.sin(global_angle)
        return x, y


    def with_uncertainty(self, sigma_distance: float, sigma_angle: float) -> "MeasurementDTO":
        noisy_distance = max(0.0, np.random.normal(self.distance, sigma_distance))
        noisy_angle = np.random.normal(self.angle, sigma_angle)

        return MeasurementDTO(noisy_distance, noisy_angle, self.hit, self.agent)


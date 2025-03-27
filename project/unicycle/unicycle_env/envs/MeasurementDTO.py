from dataclasses import dataclass
import numpy as np


@dataclass
class MeasurementDTO:
    distance: float
    angle: float
    position: tuple[float, float]

    def to_cartesian(self) -> tuple[float, float]:
        x = self.position[0] + self.distance * np.cos(self.angle)
        y = self.position[1] - self.distance * np.sin(self.angle)
        return x, y


    def with_uncertainty(self, sigma_distance: float, sigma_angle: float) -> "MeasurementDTO":
        noisy_distance = max(0.0, np.random.normal(self.distance, sigma_distance))
        noisy_angle = np.random.normal(self.angle, sigma_angle)

        return MeasurementDTO(noisy_distance, noisy_angle, self.position)


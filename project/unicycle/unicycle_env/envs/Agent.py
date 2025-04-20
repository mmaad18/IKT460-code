import math
from dataclasses import dataclass

import numpy as np
from pygame.color import Color


@dataclass
class Agent:
    position: tuple[float, float]
    angle: float
    size: tuple[float, float]
    color: Color


    def apply_action(self, action: np.ndarray, dt: float):
        v, omega = float(action[0]), float(action[1])
        x, y = self.position
        theta = self.angle

        new_x = x + v * np.cos(theta) * dt
        new_y = y - v * np.sin(theta) * dt
        new_theta = theta + omega * dt

        self.position = (new_x, new_y)
        self.angle = new_theta % (2 * np.pi)


    def get_polygon(self) -> list[tuple[int, int]]:
        x, y = self.position
        length, width = self.size
        pi_2 = math.pi / 2.0

        tip = (x + length * math.cos(self.angle), y - length * math.sin(self.angle))

        base_left = (
            x + width * 0.5 * math.cos(self.angle + pi_2),
            y - width * 0.5 * math.sin(self.angle + pi_2),
        )
        base_right = (
            x + width * 0.5 * math.cos(self.angle - pi_2),
            y - width * 0.5 * math.sin(self.angle - pi_2),
        )

        return [
            tuple(map(round, tip)),
            tuple(map(round, base_left)),
            tuple(map(round, base_right)),
        ]


    def get_pose(self) -> tuple[float, float, float]:
        return self.position[0], self.position[1], self.angle


    def get_pose_noisy(self, sigma_position: float, sigma_angle: float) -> tuple[float, float, float]:
        noisy_x = max(0.0, np.random.normal(self.position[0], sigma_position))
        noisy_y = max(0.0, np.random.normal(self.position[1], sigma_position))
        noisy_angle = np.random.normal(self.angle, sigma_angle) % (2 * np.pi)

        return noisy_x, noisy_y, noisy_angle


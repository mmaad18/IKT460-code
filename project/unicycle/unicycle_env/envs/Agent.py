import math
import random
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pygame.color import Color


@dataclass
class Agent:
    position: tuple[float, float]
    angle: float
    size: tuple[float, float] = (25.0, 20.0)
    color: Color = field(init=False)
    velocity: float = 0.0
    omega: float = 0.0
    v_damp: Final = 0.5
    omega_damp: Final = 0.75
    v_max = 250.0
    v_min = -50.0
    omega_max = 5.0


    def __post_init__(self) -> None:
        self.color = Color("green")
    
    
    def reset(self) -> None:
        self.angle = random.uniform(0, 2 * np.pi)
        self.velocity = 0.0
        self.omega = 0.0


    def apply_action(self, action: NDArray[np.float32], dt: float) -> None:
        a, alpha = action
        
        # Damping
        self.velocity *= (1.0 - self.v_damp * dt)
        self.omega *= (1.0 - self.omega_damp * dt)

        self.velocity += a * dt
        self.omega += alpha * dt
        
        self.velocity = np.clip(self.velocity, self.v_min, self.v_max)
        self.omega = np.clip(self.omega, -self.omega_max, self.omega_max)
        
        x, y = self.position
        theta = self.angle

        new_x: float = x + self.velocity * np.cos(theta) * dt
        new_y: float = y - self.velocity * np.sin(theta) * dt
        new_theta = theta + self.omega * dt

        self.position = (new_x, new_y)
        self.angle = new_theta % (2 * np.pi)


    def get_polygon(self) -> list[tuple[int, int]]:
        x, y = self.position
        length, width = self.size
        pi_2: Final = math.pi / 2.0

        tip = round((x + length * math.cos(self.angle))), round(y - length * math.sin(self.angle))

        base_left = (
            round(x + width * 0.5 * math.cos(self.angle + pi_2)),
            round(y - width * 0.5 * math.sin(self.angle + pi_2)),
        )
        base_right = (
            round(x + width * 0.5 * math.cos(self.angle - pi_2)),
            round(y - width * 0.5 * math.sin(self.angle - pi_2)),
        )

        return [
            tip,
            base_left,
            base_right,
        ]


    def get_sweepers(self) -> list[tuple[int, int]]:
        x, y = self.position
        width = self.size[1]
        pi_2: Final = math.pi / 2.0

        left_sweeper = (
            round(x + width * 0.25 * math.cos(self.angle + pi_2)),
            round(y - width * 0.25 * math.sin(self.angle + pi_2)),
        )
        right_sweeper = (
            round(x + width * 0.25 * math.cos(self.angle - pi_2)),
            round(y - width * 0.25 * math.sin(self.angle - pi_2)),
        )

        return [left_sweeper, right_sweeper]


    def get_pose(self) -> tuple[float, float, float]:
        return self.position[0], self.position[1], self.angle


    def get_pose_noisy(self, sigma_position: float, sigma_angle: float) -> tuple[float, float, float]:
        noisy_x = max(0.0, np.random.normal(self.position[0], sigma_position))
        noisy_y = max(0.0, np.random.normal(self.position[1], sigma_position))
        noisy_angle = np.random.normal(self.angle, sigma_angle) % (2 * np.pi)

        return noisy_x, noisy_y, noisy_angle


    def get_local_velocity(self) -> tuple[float, float, float]:
        return self.velocity, 0.0, self.omega


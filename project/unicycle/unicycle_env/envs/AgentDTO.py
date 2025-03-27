from dataclasses import dataclass
from pygame.color import Color
import math


@dataclass
class AgentDTO:
    position: tuple[float, float]
    angle: float
    size: tuple[float, float]
    color: Color

    def get_polygon(self) -> list[tuple[int, int]]:
        x, y = self.position
        length, width = self.size
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        pi_2 = math.pi / 2.0

        tip = (x + length * cos_a, y - length * sin_a)

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


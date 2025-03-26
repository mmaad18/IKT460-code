from dataclasses import dataclass
from pygame.color import Color


@dataclass
class ObstacleDTO:
    position: tuple[float, float]
    size: tuple[float, float]
    color: Color


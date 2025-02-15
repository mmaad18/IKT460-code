from enum import Enum


class CellType(Enum):
    START = "S"
    PATH = "·"
    OBSTACLE = "#"
    TERMINAL = "T"

    def __str__(self) -> str:
        return self.value

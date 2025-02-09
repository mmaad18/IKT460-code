from enum import Enum


class Action(Enum):
    UP = (-1, 0), "↑"
    DOWN = (1, 0), "↓"
    LEFT = (0, -1), "←"
    RIGHT = (0, 1), "→"

    def move(self):
        return self.value[0]

    def __str__(self):
        return self.value[1]
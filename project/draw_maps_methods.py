import pygame
from pygame import Surface


"""
ROOMS
"""
def draw_room(surface: Surface, origin: tuple[int, int], size: tuple[int, int], exits: str = "0000", door_width: int = 60, wall_thickness: int = 10, wall_color=(0, 0, 0)) -> None:
    """
    exits: A string of 4 characters (0 or 1) representing the exits: Top, Right, Bottom, Left.
    """
    x, y = origin
    width, height = size

    if len(exits) != 4 or any(c not in "01" for c in exits):
        raise ValueError("exits must be a 4-character string of 0 or 1")

    top_exit, right_exit, bottom_exit, left_exit = exits

    walls = []

    # TOP wall
    if top_exit == "0":
        walls.append(pygame.Rect(x, y, width, wall_thickness))
    else:
        walls.append(pygame.Rect(x, y, (width - door_width) // 2, wall_thickness))
        walls.append(pygame.Rect(x + (width + door_width) // 2, y, (width - door_width) // 2, wall_thickness))

    # RIGHT wall
    if right_exit == "0":
        walls.append(pygame.Rect(x + width - wall_thickness, y, wall_thickness, height))
    else:
        walls.append(pygame.Rect(x + width - wall_thickness, y, wall_thickness, (height - door_width) // 2))
        walls.append(pygame.Rect(x + width - wall_thickness, y + (height + door_width) // 2, wall_thickness, (height - door_width) // 2))

    # BOTTOM wall
    if bottom_exit == "0":
        walls.append(pygame.Rect(x, y + height - wall_thickness, width, wall_thickness))
    else:
        walls.append(pygame.Rect(x, y + height - wall_thickness, (width - door_width) // 2, wall_thickness))
        walls.append(pygame.Rect(x + (width + door_width) // 2, y + height - wall_thickness, (width - door_width) // 2, wall_thickness))

    # LEFT wall
    if left_exit == "0":
        walls.append(pygame.Rect(x, y, wall_thickness, height))
    else:
        walls.append(pygame.Rect(x, y, wall_thickness, (height - door_width) // 2))
        walls.append(pygame.Rect(x, y + (height + door_width) // 2, wall_thickness, (height - door_width) // 2))

    for wall in walls:
        pygame.draw.rect(surface, wall_color, wall)

"""
MAPS
"""
def draw_map_1(surface: Surface) -> None:
    draw_room(surface, (200, 160), (800, 400), "1101")
    draw_room(surface, (300, 260), (600, 300), "1101")
    draw_room(surface, (400, 360), (400, 200), "1101")
    draw_room(surface, (500, 460), (200, 100), "1101")

def draw_map_2(surface: Surface) -> None:
    draw_room(surface, (100, 100), (250, 150), "1110")
    draw_room(surface, (450, 100), (150, 150), "1011")
    draw_room(surface, (700, 100), (150, 150), "1100")
    draw_room(surface, (300, 350), (200, 150), "0011")
    draw_room(surface, (600, 400), (150, 100), "1001")

def draw_map_3(surface: Surface) -> None:
    draw_room(surface, (50, 50), (200, 150), "1111")
    draw_room(surface, (350, 50), (200, 150), "1101")
    draw_room(surface, (650, 50), (200, 150), "1011")
    draw_room(surface, (200, 300), (250, 150), "0011")
    draw_room(surface, (550, 300), (250, 150), "0011")

def draw_map_4(surface: Surface) -> None:
    draw_room(surface, (100, 50), (200, 150), "1000")
    draw_room(surface, (350, 50), (200, 150), "0010")
    draw_room(surface, (600, 50), (200, 150), "0001")
    draw_room(surface, (250, 300), (200, 150), "1111")
    draw_room(surface, (550, 300), (200, 150), "0110")

def draw_map_5(surface: Surface) -> None:
    draw_room(surface, (60, 60), (250, 150), "1010")
    draw_room(surface, (360, 60), (250, 150), "0011")
    draw_room(surface, (660, 60), (200, 150), "1010")
    draw_room(surface, (260, 360), (250, 150), "0101")
    draw_room(surface, (560, 360), (200, 150), "0010")

def draw_map_6(surface: Surface) -> None:
    draw_room(surface, (50, 100), (200, 100), "1100")
    draw_room(surface, (300, 100), (200, 100), "0110")
    draw_room(surface, (600, 100), (200, 100), "0101")
    draw_room(surface, (150, 300), (250, 100), "0010")
    draw_room(surface, (500, 300), (250, 100), "1000")

def draw_map_7(surface: Surface) -> None:
    draw_room(surface, (100, 100), (150, 150), "1010")
    draw_room(surface, (350, 100), (150, 150), "1101")
    draw_room(surface, (600, 100), (150, 150), "0011")
    draw_room(surface, (100, 400), (200, 100), "1010")
    draw_room(surface, (400, 400), (200, 100), "0011")

def draw_map_8(surface: Surface) -> None:
    draw_room(surface, (40, 40), (200, 150), "0110")
    draw_room(surface, (300, 40), (200, 150), "1101")
    draw_room(surface, (560, 40), (200, 150), "1001")
    draw_room(surface, (200, 300), (200, 150), "0110")
    draw_room(surface, (500, 300), (200, 150), "0011")

def draw_map_9(surface: Surface) -> None:
    draw_room(surface, (60, 60), (250, 150), "1011")
    draw_room(surface, (400, 60), (250, 150), "1001")
    draw_room(surface, (700, 60), (200, 150), "0110")
    draw_room(surface, (300, 300), (200, 150), "0011")
    draw_room(surface, (600, 300), (200, 150), "0010")

def draw_map_10(surface: Surface) -> None:
    draw_room(surface, (80, 80), (200, 100), "0010")
    draw_room(surface, (350, 80), (200, 100), "0010")
    draw_room(surface, (620, 80), (200, 100), "0110")
    draw_room(surface, (200, 350), (200, 100), "1001")
    draw_room(surface, (470, 350), (200, 100), "1100")

def draw_map_11(surface: Surface) -> None:
    draw_room(surface, (60, 60), (200, 100), "1010")
    draw_room(surface, (300, 60), (200, 100), "0011")
    draw_room(surface, (540, 60), (200, 100), "1011")
    draw_room(surface, (180, 240), (250, 150), "0011")
    draw_room(surface, (500, 240), (250, 150), "0011")

def draw_map_12(surface: Surface) -> None:
    draw_room(surface, (100, 100), (200, 150), "1110")
    draw_room(surface, (360, 100), (200, 150), "1011")
    draw_room(surface, (620, 100), (200, 150), "1100")
    draw_room(surface, (250, 350), (200, 100), "1010")
    draw_room(surface, (500, 350), (200, 100), "0011")

def draw_map_13(surface: Surface) -> None:
    draw_room(surface, (100, 100), (200, 150), "1110")
    draw_room(surface, (360, 100), (200, 150), "1011")
    draw_room(surface, (620, 100), (200, 150), "1100")
    draw_room(surface, (250, 350), (200, 100), "1010")
    draw_room(surface, (500, 350), (200, 100), "0011")


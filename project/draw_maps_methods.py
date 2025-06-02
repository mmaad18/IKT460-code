import pygame
from pygame import Surface, Color


"""
ROOMS
"""
def draw_room(surface: Surface, origin: tuple[int, int], size: tuple[int, int], exits: str = "0000", door_width: int = 60, wall_thickness: int = 10, wall_color: Color = Color(0, 0, 0)) -> None:
    """
    exits: A string of 4 characters (0 or 1) representing the exits: Top, Right, Bottom, Left.
    """
    x, y = origin
    width, height = size

    if len(exits) != 4 or any(c not in "01" for c in exits):
        raise ValueError("exits must be a 4-character string of 0 or 1")

    top_exit, right_exit, bottom_exit, left_exit = exits

    walls: list[pygame.Rect] = []

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
    room_size = (190, 100)
    spacing_x = 60
    spacing_y = 60
    cols = 4
    origin_x, origin_y = 130, 90

    exits = [
        "0110", "0111", "0111", "0011",
        "1110", "1011", "1110", "1011",
        "1100", "1101", "1101", "1001",
    ]

    for i in range(12):
        row = i // cols
        col = i % cols

        x = origin_x + col * (room_size[0] + spacing_x)
        y = origin_y + row * (room_size[1] + spacing_y)

        draw_room(surface, (x, y), room_size, exits[i])

def draw_map_3(surface: Surface) -> None:
    draw_room(surface, (110, 110), (980, 380), "0001")
    draw_room(surface, (180, 180), (840, 240), "0100")
    draw_room(surface, (250, 250), (700, 100), "0001")

def draw_map_4(surface: Surface) -> None:
    draw_room(surface, (200, 110), (400, 200), "1001", door_width=80)
    draw_room(surface, (600, 110), (400, 200), "1100", door_width=80)
    draw_room(surface, (600, 290), (400, 200), "0110", door_width=80)
    draw_room(surface, (200, 290), (400, 200), "0011", door_width=80)

def draw_map_5(surface: Surface) -> None:
    draw_room(surface, (40, 40), (1120, 180), "0010", door_width=100)
    draw_room(surface, (40, 210), (1120, 180), "1010", door_width=100)
    draw_room(surface, (40, 380), (1120, 180), "1000", door_width=100)

def draw_map_6(surface: Surface) -> None:
    draw_room(surface, (40, 40), (190, 520), "0100", door_width=120)
    draw_room(surface, (220, 40), (190, 520), "0101", door_width=120)
    draw_room(surface, (400, 40), (190, 520), "0101", door_width=120)
    draw_room(surface, (580, 40), (190, 520), "0101", door_width=120)
    draw_room(surface, (760, 40), (190, 520), "0101", door_width=120)

def draw_map_7(surface: Surface) -> None:
    draw_room(surface, (40, 40), (360, 140), "0010")
    draw_room(surface, (800, 40), (360, 140), "0010")
    draw_room(surface, (420, 230), (360, 140), "1010")
    draw_room(surface, (40, 420), (360, 140), "1000")
    draw_room(surface, (800, 420), (360, 140), "1000")

def draw_map_8(surface: Surface) -> None:
    room_size = (70, 70)
    spacing_x = 60
    spacing_y = 60
    cols = 8
    origin_x, origin_y = 110, 70

    for i in range(32):
        row = i // cols
        col = i % cols

        x = origin_x + col * (room_size[0] + spacing_x)
        y = origin_y + row * (room_size[1] + spacing_y)

        draw_room(surface, (x, y), room_size, "0000")

def draw_map_9(surface: Surface) -> None:
    draw_room(surface, (40, 40), (1120, 520), "0000")

def draw_map_10(surface: Surface) -> None:
    draw_room(surface, (40, 130), (140, 140), "0100")
    draw_room(surface, (280, 350), (250, 210), "1000", door_width=100)
    draw_room(surface, (650, 320), (400, 100), "0101")
    draw_room(surface, (700, 40), (200, 150), "0010")

def draw_map_11(surface: Surface) -> None:
    room_size = (380, 180)
    spacing_x = -10
    spacing_y = -10
    cols = 3
    origin_x, origin_y = 40, 40

    exits = [
        "0110", "0101", "0001",
        "1100", "0101", "0011",
        "0100", "0101", "1001",
    ]

    for i in range(9):
        row = i // cols
        col = i % cols

        x = origin_x + col * (room_size[0] + spacing_x)
        y = origin_y + row * (room_size[1] + spacing_y)

        draw_room(surface, (x, y), room_size, exits[i], door_width=100)

def draw_map_12(surface: Surface) -> None:
    draw_room(surface, (120, 120), (200, 150), "1110")
    draw_room(surface, (700, 170), (350, 150), "1011")
    draw_room(surface, (400, 340), (200, 150), "1100")

def draw_map_13(surface: Surface) -> None:
    draw_room(surface, (40, 40), (160, 160), "1111")
    draw_room(surface, (1000, 40), (160, 160), "1111")
    draw_room(surface, (40, 400), (160, 160), "1111")
    draw_room(surface, (1000, 400), (160, 160), "1111")
    draw_room(surface, (300, 150), (600, 300), "1111", door_width=200)





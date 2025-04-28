import pygame
from pygame import Surface

"""
GRID
"""
def draw_grid(surface: Surface, map_size=(1200, 600)):
    grid_spacing = 10
    grid_color = (200, 200, 200)

    # Draw vertical grid lines
    for x in range(0, map_size[0], grid_spacing):
        pygame.draw.line(surface, grid_color, (x, 0), (x, map_size[1]), width=1)

    # Draw horizontal grid lines
    for y in range(0, map_size[1], grid_spacing):
        pygame.draw.line(surface, grid_color, (0, y), (map_size[0], y), width=1)


"""
OUTER BOX
"""
def draw_outer_box(surface: Surface, map_size=(1200, 600), wall_thickness=10, wall_color=(0, 0, 0)):
    outer_box = [
        # Outer box
        pygame.Rect(40, 40, map_size[0] - 80, wall_thickness), # Top
        pygame.Rect(40, map_size[1] - wall_thickness - 40, map_size[0] - 80, wall_thickness), # Bottom
        pygame.Rect(40, 40, wall_thickness, map_size[1] - 80), # Left
        pygame.Rect(map_size[0] - wall_thickness - 40, 40, wall_thickness, map_size[1] - 80), # Right
    ]

    for wall in outer_box:
        pygame.draw.rect(surface, wall_color, wall)


"""
ROOMS
"""
def draw_room(surface: Surface, origin: tuple[int, int], size: tuple[int, int], exits: str = "0000", door_width: int = 60, wall_thickness: int = 10, wall_color=(0, 0, 0)):
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
def draw_map_1(surface: Surface):
    draw_room(surface, (40, 40), (200, 100), "0010")
    draw_room(surface, (300, 40), (200, 100), "1010")
    draw_room(surface, (600, 40), (200, 100), "1001")
    draw_room(surface, (300, 250), (200, 150), "1100")
    draw_room(surface, (600, 250), (200, 150), "0101")
    draw_room(surface, (200, 450), (250, 100), "0011")

def draw_map_2(surface: Surface):
    draw_room(surface, (100, 100), (250, 150), "1110")
    draw_room(surface, (450, 100), (150, 150), "1011")
    draw_room(surface, (700, 100), (150, 150), "1100")
    draw_room(surface, (300, 350), (200, 150), "0011")
    draw_room(surface, (600, 400), (150, 100), "1001")

def draw_map_3(surface: Surface):
    draw_room(surface, (50, 50), (200, 150), "1111")
    draw_room(surface, (350, 50), (200, 150), "1101")
    draw_room(surface, (650, 50), (200, 150), "1011")
    draw_room(surface, (200, 300), (250, 150), "0011")
    draw_room(surface, (550, 300), (250, 150), "0011")

def draw_map_4(surface: Surface):
    draw_room(surface, (100, 50), (200, 150), "1000")
    draw_room(surface, (350, 50), (200, 150), "0010")
    draw_room(surface, (600, 50), (200, 150), "0001")
    draw_room(surface, (250, 300), (200, 150), "1111")
    draw_room(surface, (550, 300), (200, 150), "0110")

def draw_map_5(surface: Surface):
    draw_room(surface, (60, 60), (250, 150), "1010")
    draw_room(surface, (360, 60), (250, 150), "0011")
    draw_room(surface, (660, 60), (200, 150), "1010")
    draw_room(surface, (260, 360), (250, 150), "0101")
    draw_room(surface, (560, 360), (200, 150), "0010")

def draw_map_6(surface: Surface):
    draw_room(surface, (50, 100), (200, 100), "1100")
    draw_room(surface, (300, 100), (200, 100), "0110")
    draw_room(surface, (600, 100), (200, 100), "0101")
    draw_room(surface, (150, 300), (250, 100), "0010")
    draw_room(surface, (500, 300), (250, 100), "1000")

def draw_map_7(surface: Surface):
    draw_room(surface, (100, 100), (150, 150), "1010")
    draw_room(surface, (350, 100), (150, 150), "1101")
    draw_room(surface, (600, 100), (150, 150), "0011")
    draw_room(surface, (100, 400), (200, 100), "1010")
    draw_room(surface, (400, 400), (200, 100), "0011")

def draw_map_8(surface: Surface):
    draw_room(surface, (40, 40), (200, 150), "0110")
    draw_room(surface, (300, 40), (200, 150), "1101")
    draw_room(surface, (560, 40), (200, 150), "1001")
    draw_room(surface, (200, 300), (200, 150), "0110")
    draw_room(surface, (500, 300), (200, 150), "0011")

def draw_map_9(surface: Surface):
    draw_room(surface, (60, 60), (250, 150), "1011")
    draw_room(surface, (400, 60), (250, 150), "1001")
    draw_room(surface, (700, 60), (200, 150), "0110")
    draw_room(surface, (300, 300), (200, 150), "0011")
    draw_room(surface, (600, 300), (200, 150), "0010")

def draw_map_10(surface: Surface):
    draw_room(surface, (80, 80), (200, 100), "0010")
    draw_room(surface, (350, 80), (200, 100), "0010")
    draw_room(surface, (620, 80), (200, 100), "0110")
    draw_room(surface, (200, 350), (200, 100), "1001")
    draw_room(surface, (470, 350), (200, 100), "1100")


"""
MAIN
"""
def main():
    map_size = (1200, 600)
    bg_color = (255, 255, 255)
    map_functions = [
        draw_map_1, draw_map_2, draw_map_3, draw_map_4, draw_map_5,
        draw_map_6, draw_map_7, draw_map_8, draw_map_9, draw_map_10
    ]

    pygame.init()

    for idx, draw_map_fn in enumerate(map_functions, start=1):
        surface = pygame.Surface(map_size)
        surface.fill(bg_color)

        draw_grid(surface)
        draw_outer_box(surface)
        draw_map_fn(surface)

        pygame.image.save(surface, f"generated_maps/generated_map_{idx}.png")

    pygame.quit()


main()



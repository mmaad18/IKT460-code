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
    draw_room(surface, (40, 40), (200, 100), "0000")
    draw_room(surface, (300, 400), (200, 100), "0110")
    draw_room(surface, (500, 200), (100, 100), "1111")
    draw_room(surface, (700, 300), (200, 100), "0110")


def main():
    # Map setup
    map_size = (1200, 600)
    bg_color = (255, 255, 255)

    pygame.init()
    surface = pygame.Surface(map_size)
    surface.fill(bg_color)

    # Draw the grid
    draw_grid(surface)

    # Draw the walls
    draw_outer_box(surface)
    draw_map_1(surface)

    # Save the surface to a file
    pygame.image.save(surface, "generated_map.png")
    pygame.quit()


main()



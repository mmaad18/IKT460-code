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
def draw_room_1(surface: Surface, wall_thickness=10, wall_color=(0, 0, 0)):
    room_1 = [
        pygame.Rect(690, 200, 80, wall_thickness), # Bottom Left
        pygame.Rect(830, 200, 80, wall_thickness), # Bottom Right
        pygame.Rect(690, 40, wall_thickness, 170), # Left
        pygame.Rect(900, 40, wall_thickness, 170), # Right
    ]

    for wall in room_1:
        pygame.draw.rect(surface, wall_color, wall)


def draw_room_2(surface: Surface, wall_thickness=10, wall_color=(0, 0, 0)):
    room_2 = [
        pygame.Rect(190, 390, 80, wall_thickness), # Top Left
        pygame.Rect(330, 390, 80, wall_thickness), # Top Right
        pygame.Rect(190, 390, wall_thickness, 170), # Left
        pygame.Rect(400, 390, wall_thickness, 170), # Right
    ]

    for wall in room_2:
        pygame.draw.rect(surface, wall_color, wall)


def draw_room(surface: Surface, origin_x: int, origin_y: int, wall_thickness=10, wall_color=(0, 0, 0)):
    walls = [
        pygame.Rect(origin_x, origin_y + 160, 80, wall_thickness),  # Bottom Left door
        pygame.Rect(origin_x + 140, origin_y + 160, 80, wall_thickness),  # Bottom Right door
        pygame.Rect(origin_x, origin_y, wall_thickness, 170),  # Left wall
        pygame.Rect(origin_x + 210, origin_y, wall_thickness, 170),  # Right wall
    ]

    for wall in walls:
        pygame.draw.rect(surface, wall_color, wall)



def draw_tunnel_1(surface: Surface, wall_thickness=10, wall_color=(0, 0, 0)):
    tunnel_1 = [
        pygame.Rect(590, 340, 420, wall_thickness), # Top
        pygame.Rect(590, 450, 420, wall_thickness), # Bottom
        pygame.Rect(590, 340, wall_thickness, 30), # Left Top
        pygame.Rect(590, 430, wall_thickness, 30), # Left Bottom
        pygame.Rect(1000, 340, wall_thickness, 30), # Right Top
        pygame.Rect(1000, 430, wall_thickness, 30), # Right Bottom
    ]

    for wall in tunnel_1:
        pygame.draw.rect(surface, wall_color, wall)


"""
WALLS
"""
def draw_walls(surface: Surface, wall_thickness=10, wall_color=(0, 0, 0)):
    walls = [
        # Room 1
        pygame.Rect(690, 200, 80, wall_thickness), # Bottom Left
        pygame.Rect(830, 200, 80, wall_thickness), # Bottom Right
        pygame.Rect(690, 40, wall_thickness, 170), # Left
        pygame.Rect(900, 40, wall_thickness, 170), # Right

        # Room 2
        pygame.Rect(190, 390, 80, wall_thickness), # Top Left
        pygame.Rect(330, 390, 80, wall_thickness), # Top Right
        pygame.Rect(190, 390, wall_thickness, 170), # Left
        pygame.Rect(400, 390, wall_thickness, 170), # Right

        # Tunnel 1
        pygame.Rect(590, 340, 420, wall_thickness), # Top
        pygame.Rect(590, 450, 420, wall_thickness), # Bottom
        pygame.Rect(590, 340, wall_thickness, 30), # Left Top
        pygame.Rect(590, 430, wall_thickness, 30), # Left Bottom
        pygame.Rect(1000, 340, wall_thickness, 30), # Right Top
        pygame.Rect(1000, 430, wall_thickness, 30), # Right Bottom

        # Tunnel 2
        pygame.Rect(40, 140, 120, wall_thickness), # Top
        pygame.Rect(40, 250, 120, wall_thickness), # Bottom
        pygame.Rect(150, 140, wall_thickness, 30), # Right Top
        pygame.Rect(150, 230, wall_thickness, 30), # Right Bottom
    ]

    for wall in walls:
        pygame.draw.rect(surface, wall_color, wall)


def main():
    # Map setup
    map_size = (1200, 600)
    bg_color = (255, 255, 255)
    wall_color = (0, 0, 0)
    wall_thickness = 10

    pygame.init()
    surface = pygame.Surface(map_size)
    surface.fill(bg_color)

    # Draw the grid
    draw_grid(surface)

    # Draw the walls
    draw_outer_box(surface)
    draw_room(surface, 60, 40)
    draw_tunnel_1(surface)
    #draw_walls(surface)

    # Save the surface to a file
    pygame.image.save(surface, "generated_map.png")
    pygame.quit()


main()



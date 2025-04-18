import pygame


# Map setup
map_size = (1200, 600)
bg_color = (255, 255, 255)
wall_color = (0, 0, 0)
wall_thickness = 10

pygame.init()
surface = pygame.Surface(map_size)
surface.fill(bg_color)

"""
GRID
"""
def draw_grid():
    grid_spacing = 50  # change as needed
    grid_color = (200, 200, 200)  # light gray

    # Draw vertical grid lines
    for x in range(0, map_size[0], grid_spacing):
        pygame.draw.line(surface, grid_color, (x, 0), (x, map_size[1]), width=1)

    # Draw horizontal grid lines
    for y in range(0, map_size[1], grid_spacing):
        pygame.draw.line(surface, grid_color, (0, y), (map_size[0], y), width=1)


"""
WALLS
"""
def draw_walls():
    walls = [
        # Outer box
        pygame.Rect(40, 40, 1120, wall_thickness), # Top
        pygame.Rect(40, 550, 1120, wall_thickness), # Bottom
        pygame.Rect(40, 40, wall_thickness, 520), # Left
        pygame.Rect(1150, 40, wall_thickness, 520), # Right

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


"""
Save to PNG
"""
def save_surface():
    pygame.image.save(surface, "generated_map.png")
    pygame.quit()


draw_grid()
draw_walls()
save_surface()



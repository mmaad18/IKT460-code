import pygame
from pygame import Surface
import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread, imsave
from tqdm import tqdm

from project.draw_maps_methods import *

"""
GRID
"""
def draw_grid(surface: Surface, map_size=(1200, 600)) -> None:
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
def draw_outer_box(surface: Surface, map_size=(1200, 600), wall_thickness=10, wall_color=(0, 0, 0)) -> None:
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
STARTING POSITION
"""
def generate_start_map_and_positions(image_read_path: str, image_write_path: str, start_positions_path: str, margin: int = 75, kernel_size: int = 51) -> None:
    image_array = imread(image_read_path)
    bw_image = np.dot(image_array[..., :3], np.array([1/3, 1/3, 1/3]))
    obstacle_map = bw_image < 128
    start_map = ~(convolve2d(obstacle_map, np.ones((kernel_size, kernel_size)), mode='same') > 0)

    imsave(image_write_path, (start_map * 255).astype(np.uint8))

    start_positions: list[tuple[int, int]] = []
    h, w = start_map.shape

    for x in range(margin, w - margin):
        for y in range(margin, h - margin):
            if start_map[y, x]:
                start_positions.append((x, y))

    np.random.shuffle(start_positions)
    np.save(start_positions_path, np.array(start_positions, dtype=np.int32))


"""
MAIN
"""
def main() -> None:
    map_size = (1200, 600)
    bg_color = (255, 255, 255)
    map_functions = [
        draw_map_1, draw_map_2, draw_map_3, draw_map_4, draw_map_5,
        #draw_map_6, draw_map_7, draw_map_8, draw_map_9, draw_map_10,
        #draw_map_11, draw_map_12, draw_map_13
    ]

    pygame.init()

    for idx, draw_map_fn in tqdm(enumerate(map_functions, start=1), total=len(map_functions), desc="Generating maps"):
        surface = pygame.Surface(map_size)
        surface.fill(bg_color)

        draw_grid(surface)
        draw_outer_box(surface)
        draw_map_fn(surface)

        map_path = f"generated/maps/map_{idx}.png"
        start_map_path = f"generated/start_maps/map_{idx}.png"
        positions_path = f"generated/start_positions/map_{idx}.npy"

        pygame.image.save(surface, map_path)
        generate_start_map_and_positions(map_path, start_map_path, positions_path)

    pygame.quit()


main()



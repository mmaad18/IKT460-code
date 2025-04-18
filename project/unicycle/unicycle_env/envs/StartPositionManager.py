import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread


class StartPositionManager:
    def __init__(self, map_image_path: str, margin: int = 75, kernel_size: int = 51):
        self.margin = margin
        self.kernel_size = kernel_size
        self.map_image_path = map_image_path

        self.idx = 0
        self.starting_positions_count = -1
        self.starting_positions: list[tuple[int, int]] = []

        self._load_starting_positions()


    def next(self) -> tuple[int, int]:
        self.idx = (self.idx + 1) % self.starting_positions_count
        return self.starting_positions[self.idx]


    def _load_starting_positions(self):
        image_array = imread(self.map_image_path)
        bw_image = np.dot(image_array[..., :3], np.array([1/3, 1/3, 1/3]))
        obstacle_map = bw_image < 128
        dilated_obstacle_map = ~(convolve2d(obstacle_map, np.ones((self.kernel_size, self.kernel_size)), mode='same') > 0)

        self.starting_positions.clear()
        h, w = dilated_obstacle_map.shape

        for x in range(self.margin, w - self.margin):
            for y in range(self.margin, h - self.margin):
                if dilated_obstacle_map[y, x]:
                    self.starting_positions.append((x, y))

        np.random.shuffle(self.starting_positions)
        self.starting_positions_count = len(self.starting_positions)




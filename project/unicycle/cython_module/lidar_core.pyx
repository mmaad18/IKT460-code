# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False
import cython
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, fmod, M_PI

@cython.boundscheck(False)
@cython.wraparound(False)
def lidar_measurement(
        np.ndarray[np.float32_t, ndim=2] measurements,
        np.ndarray[np.float64_t, ndim=1] relative_angles,
        float agent_x, float agent_y, float agent_angle,
        int width, int height,
        int max_distance, int step,
        np.ndarray[np.uint8_t, ndim=2] obstacle_map
):

    cdef int i, distance, x2, y2
    cdef float global_angle, cos_a, sin_a
    cdef float hit
    cdef float x1 = agent_x
    cdef float y1 = agent_y

    for i in range(relative_angles.shape[0]):
        global_angle = fmod(agent_angle + relative_angles[i], 2 * M_PI)
        cos_a = cos(global_angle)
        sin_a = sin(global_angle)
        hit = 0.0

        for distance in range(0, max_distance, step):
            x2 = <int>(x1 + distance * cos_a)
            y2 = <int>(y1 - distance * sin_a)

            if 0 <= x2 < width and 0 <= y2 < height:
                if obstacle_map[x2, y2]:  # Assuming boolean NumPy map
                    hit = 1.0
                    measurements[i, 0] = distance
                    measurements[i, 1] = relative_angles[i]
                    measurements[i, 2] = hit
                    measurements[i, 3] = x2
                    measurements[i, 4] = y2
                    break

        if hit == 0.0:
            x2 = <int>(x1 + max_distance * cos_a)
            y2 = <int>(y1 - max_distance * sin_a)
            measurements[i, 0] = max_distance
            measurements[i, 1] = relative_angles[i]
            measurements[i, 2] = hit
            measurements[i, 3] = x2
            measurements[i, 4] = y2


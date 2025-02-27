from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 512
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32),
            np.array([+1, +1, +1]).astype(np.float32),
        )  # steer, gas, brake

        self.observation_space = spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        pass



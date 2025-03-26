from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType

from unicycle_env.envs.AgentDTO import AgentDTO


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        from project.unicycle.unicycle_env.envs import LidarEnvironment
        from project.unicycle.unicycle_env.envs import Lidar

        self.render_mode = render_mode
        self.clock = None
        self.window = None
        self.window_size = 512
        self.map_path = "project/unicycle/unicycle_env/envs/SLAM_MAP_1H.png"
        self.map_dimensions = (600, 1200)

        self.environment = LidarEnvironment.LidarEnvironment(self.map_path, self.map_dimensions)
        self.lidar = Lidar.Lidar(self.environment, max_distance=100, uncertianty=(5, 0.1))

        self.agente = AgentDTO(position=(100.0, 100.0), angle=0.0, size=(20, 10), color=pygame.Color("green"))

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=(60,), dtype=np.float32)  # one float per LIDAR ray



    def step(self, action):
        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        terminated = self._is_terminated(observation)
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state

        info = {}

        return self._get_observation(), info


    def _get_observation(self):
        # Simulate LIDAR readings or other sensor data
        return self.lidar.measurement(self.agent_position, 60)


    def _calculate_reward(self, observation):
        # Define a reward function based on the observation
        return np.sum(observation)  # Placeholder


    def _is_terminated(self, observation):
        # Define a condition to end the episode
        return np.any(observation > 0.9)  # Placeholder


    def render(self, mode='human'):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        if self.render_mode == "human":
            # Copy drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



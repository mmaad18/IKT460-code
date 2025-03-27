from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.color import Color
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType

from unicycle_env.envs.AgentDTO import AgentDTO
from unicycle_env.envs.Lidar import Lidar
from unicycle_env.envs.LidarEnvironment import LidarEnvironment
from unicycle_env.envs.MeasurementDTO import MeasurementDTO


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.clock = None
        self.window = None
        self.map_path = "project/unicycle/unicycle_env/envs/SLAM_MAP_1H.png"
        self.map_dimensions = (1200, 600)

        self.environment = LidarEnvironment(self.map_path, self.map_dimensions)
        self.lidar = Lidar(self.environment, max_distance=500, num_rays=60, uncertainty=(0.5, 0.01))

        self.agent = AgentDTO(position=(500.0, 500.0), angle=0.0, size=(20, 10), color=Color("green"))

        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=500.0, shape=(60,), dtype=np.float32)  # one float per LIDAR ray



    def step(self, action: np.ndarray):
        # Apply unicycle kinematics
        self._apply_action(action)

        # LIDAR observation
        measurements = self.lidar.measurement(self.agent.position)
        observation = np.array([m.distance for m in measurements], dtype=np.float32)

        reward = self._calculate_reward(observation)
        terminated = self._is_terminated(observation)
        info = {}

        if self.render_mode == "human":
            self._render_frame(measurements)

        return observation, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent.position = (500.0, 500.0)
        self.agent.angle = 0.0

        return self._get_observation(), {}


    def _get_observation(self):
        measurements = self.lidar.measurement(self.agent.position)
        return np.array([m.distance for m in measurements], dtype=np.float32)


    def _apply_action(self, action: np.ndarray):
        v, omega = float(action[0]), float(action[1])
        x, y = self.agent.position
        theta = self.agent.angle
        dt = 1.0 / self.metadata["render_fps"]

        new_x = x + v * np.cos(theta) * dt
        new_y = y - v * np.sin(theta) * dt  # Minus because pygame y-axis goes down
        new_theta = theta + omega * dt

        self.agent.position = (new_x, new_y)
        self.agent.angle = new_theta % (2 * np.pi)


    def _calculate_reward(self, observation):
        # Define a reward function based on the observation
        return -np.mean(observation)  # Try to keep closer to obstacles = explore


    def _is_terminated(self, observation):
        # Define a condition to end the episode
        return False


    def _render_frame(self, lidar_data: list[MeasurementDTO]):
        self.environment.update(self.agent, lidar_data)

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(self.map_dimensions)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        pygame.event.pump()

        self.window.blit(self.environment.surface, (0, 0))
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])


    def render(self, mode='human'):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



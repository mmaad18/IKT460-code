from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame.color import Color
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType

from unicycle_env.envs.AgentDTO import AgentDTO
from unicycle_env.envs.CoverageGridDTO import CoverageGridDTO
from unicycle_env.envs.Lidar import Lidar
from unicycle_env.envs.LidarEnvironment import LidarEnvironment
from unicycle_env.envs.MeasurementDTO import MeasurementDTO


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.clock = None
        self.window = None

        # Map setup
        self.map_path = "project/unicycle/unicycle_env/envs/SLAM_MAP_1H.png"
        self.map_dimensions = (1200, 600)
        self.environment = LidarEnvironment(self.map_path, self.map_dimensions)

        # Agent setup
        self.start_position = (500.0, 500.0)
        self.start_angle = 0.0
        self.num_rays = 60
        self.max_distance = 200
        self.lidar = Lidar(self.environment, max_distance=self.max_distance, num_rays=self.num_rays, uncertainty=(0.5, 0.01))
        self.agent = AgentDTO(position=self.start_position, angle=self.start_angle, size=(20, 10), color=Color("green"))

        # Action and observation space
        self.action_space = spaces.Box(low=np.array([-50.0, -5.0]), high=np.array([50.0, 5.0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=(self.max_distance+100.0), shape=(self.num_rays * 3,), dtype=np.float32)

        # Reward
        self.time_penalty = 0.01
        self.grid_resolution = 5
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        self.prev_coverage = 0
        self.collision_penalty = 10.0


    def step(self, action: np.ndarray):
        # Apply unicycle kinematics
        self._apply_action(action)

        # LIDAR observation
        measurements = self.lidar.measurement(self.agent)
        obs_2d = np.array([[m.distance, m.angle, m.hit] for m in measurements], dtype=np.float32)
        obs_flat = obs_2d.flatten()

        # Reward
        self.coverage_grid.visited(self.agent.position)
        reward = self._calculate_reward(action[0])
        terminated = self._check_collision()
        info = {}

        if self.render_mode == "human":
            self._render_frame(measurements)

        # Logging
        if True:
            print(f"  Pos: {self.agent.position}")
            print(f"  Angle: {np.degrees(self.agent.angle):.1f}°")
            print(f"  Coverage: {self.coverage_grid.coverage()}, Reward: {reward:.2f}, Terminated: {terminated}")

        return obs_flat, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent.position = self.start_position
        self.agent.angle = self.start_angle
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)

        return self._get_observation(), {}


    def _get_observation(self):
        measurements = self.lidar.measurement(self.agent)
        obs_2d = np.array([[m.distance, m.angle, m.hit] for m in measurements], dtype=np.float32)
        return obs_2d.flatten()


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

        if True:
            print(f"  Action: v={v:.2f}, omega={omega:.2f}")


    def _calculate_reward(self, velocity: float) -> float:
        current = self.coverage_grid.coverage()
        delta = current - self.prev_coverage
        self.prev_coverage = current

        reward = delta - self.time_penalty - (10 / (1 + abs(velocity))**1.5)

        if self._check_collision():
            reward -= self.collision_penalty

        return reward


    def _check_collision(self) -> bool:
        for px, py in self.agent.get_polygon():
            if self.environment.get_at((px, py)) == Color("black"):
                return True

        return False


    def _render_frame(self, lidar_data: list[MeasurementDTO]):
        self.environment.update(self.agent, self.coverage_grid, lidar_data)

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



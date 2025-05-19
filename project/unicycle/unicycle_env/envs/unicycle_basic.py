import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame.color import Color
from pathlib import Path
from tqdm import tqdm

from unicycle_env.envs.Agent import Agent
from unicycle_env.envs.CoverageGridDTO import CoverageGridDTO
from unicycle_env.envs.Lidar import Lidar
from unicycle_env.envs.LidarEnvironment import LidarEnvironment


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None) -> None:
        self.render_mode = render_mode
        self.clock = None
        self.window = None

        # Constants
        self.num_rays = 60
        self.max_distance = 250
        self.dt = 1.0 / self.metadata["render_fps"]

        # Environments setup
        self.map_dimensions = (1200, 600)
        self.environments = self._load_environments("project/generated")

        # Agent setup
        self.lidar: Lidar = None
        self.environment: LidarEnvironment = None
        self.select_environment(1)

        start_position = self.environment.next_start_position()
        self.agent = Agent(position=start_position, angle=0.0, size=(25, 16), color=Color("green"))

        # Action and observation space
        self.action_space = spaces.Box(low=np.array([-50.0, -5.0]), high=np.array([250.0, 5.0]), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=(self.max_distance+100.0), shape=(self.num_rays * 2 + 3,), dtype=np.float32)

        # Coverage grid
        self.grid_resolution = 10
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        self.prev_coverage = 0

        # Rewards
        self.time_penalty = -1.0 / self.dt
        self.omega_penalty = -0.2
        self.collision_penalty = -100.0

        self.v_reward = 0.75
        self.coverage_reward = 10.0


    def step(self, action: np.ndarray):
        # Apply unicycle kinematics
        self.agent.apply_action(action, self.dt)

        # LIDAR observation
        measurements = self.lidar.measurement(self.agent)
        obs_flat = self._get_observation(measurements)

        # Reward
        self.coverage_grid.visited(self.agent.position)
        reward = self._calculate_reward(action)
        terminated = self._check_collision()
        info = {}

        if self.render_mode == "human":
            self._render_frame(measurements)

        # Logging
        if False:
            print(f"  Action: v={float(action[0]):.2f}, omega={float(action[1]):.2f}")
            print(f"  Pos: {self.agent.position}")
            print(f"  Angle: {np.degrees(self.agent.angle):.1f}°")
            print(f"  Coverage: {self.coverage_grid.coverage()}, Reward: {reward:.2f}, Terminated: {terminated}")

        return obs_flat, reward, terminated, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent.position = self.environment.next_start_position()
        self.agent.angle = random.uniform(0, 2 * np.pi)
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        measurements = self.lidar.measurement(self.agent)

        return self._get_observation(measurements), {}


    def select_environment(self, idx: int) -> None:
        self.environment = self.environments[idx - 1]
        self.lidar = Lidar(self.environment, max_distance=self.max_distance, num_rays=self.num_rays, uncertainty=(1.0, 0.01))


    def get_environment_count(self) -> int:
        return len(self.environments)


    def _load_environments(self, base_folder: str) -> list[LidarEnvironment]:
        environments = []
        maps_folder = Path(base_folder) / "maps"
        start_positions_folder = Path(base_folder) / "start_positions"

        for map_file in tqdm(sorted(maps_folder.glob("map_*.png")), desc="Loading maps"):
            map_filename = map_file.stem
            start_positions_path = start_positions_folder / f"{map_filename}.npy"

            env = LidarEnvironment(str(map_file), str(start_positions_path), self.map_dimensions)
            environments.append(env)

        return environments


    def _get_observation(self, measurements: np.ndarray) -> np.ndarray:
        # measurements shape: (num_rays, 5) -> [distance, angle, hit, x, y]
        distances = np.clip(measurements[:, 0], 0, self.max_distance) / self.max_distance
        hits = measurements[:, 2]
        obs_2d_normalized = np.stack((distances, hits), axis=1)

        # Normalize pose
        x, y, angle = self.agent.get_pose_noisy(sigma_position=1.0, sigma_angle=0.01)
        width, height = self.map_dimensions
        pose_normalized = np.array([
            x / width,
            y / height,
            angle % (2 * np.pi) / (2 * np.pi)
        ], dtype=np.float32)

        return np.concatenate((obs_2d_normalized.ravel(), pose_normalized), axis=0)


    def _calculate_reward(self, action: np.ndarray) -> float:
        current = self.coverage_grid.coverage()
        delta = current - self.prev_coverage
        self.prev_coverage = current

        v, omega = float(action[0]), float(action[1])
        reward = self.time_penalty + self.coverage_reward * delta + self.v_reward * v + self.omega_penalty * abs(omega)

        if self._check_collision():
            reward += self.collision_penalty

        return reward


    def _check_collision(self) -> bool:
        for px, py in self.agent.get_polygon():
            if self.environment.get_at((px, py)):
                return True

        return False


    def _render_frame(self, measurements: np.ndarray) -> None:
        self.environment.update(self.agent, self.coverage_grid, measurements)

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



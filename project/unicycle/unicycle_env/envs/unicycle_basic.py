import random
from typing import Optional, Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pygame.color import Color
from pathlib import Path
from tqdm import tqdm
from numpy.typing import NDArray

from unicycle_env.envs.Agent import Agent  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.CoverageGridDTO import CoverageGridDTO  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.Lidar import Lidar  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.Imu import Imu  # pyright: ignore [reportMissingTypeStubs]
from unicycle_env.envs.LidarEnvironment import LidarEnvironment  # pyright: ignore [reportMissingTypeStubs]


class UniCycleBasicEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        self.render_mode = render_mode
        self.clock = None
        self.window = None

        # Constants
        self.num_rays = 60
        self.max_distance = 250
        self.dt: float = 1.0 / self.metadata["render_fps"]

        # Environments setup
        self.map_dimensions = (1200, 600)
        self.environments = self._load_environments("project/generated")

        # Agent setup
        self.lidar: Lidar
        self.Imu: Imu = Imu(last_pose=(0.0, 0.0, 0.0), last_velocity=(0.0, 0.0, 0.0))
        self.environment: LidarEnvironment
        self.select_environment(1)

        start_position = self.environment.next_start_position()
        self.agent = Agent(position=start_position, angle=0.0, size=(25, 20), color=Color("green"))

        self.v_max = 250.0
        self.v_min = -50.0
        self.omega_max = 5.0

        # Action and observation space
        self.action_space = spaces.Box(
            low=np.array([self.v_min, -self.omega_max]),
            high=np.array([self.v_max, self.omega_max]),
            shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=(self.max_distance+100.0),
            shape=(self.num_rays * 2 + 3,), dtype=np.float32
        )

        # Coverage grid
        self.grid_resolution = 10
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        self.prev_coverage = 0

        # Rewards
        self.time_penalty = -1.0 / self.dt
        self.omega_penalty = -0.5 / self.omega_max
        self.collision_penalty = -100.0

        self.v_reward = 10.0 / self.v_max
        self.coverage_reward = 500.0


    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # Apply unicycle kinematics
        self.agent.apply_action(action, self.dt)

        # LIDAR observation
        lidar_measurements = self.lidar.measurement(self.agent)
        imu_measurements = self.Imu.measurement(self.agent, self.dt)
        obs_flat = self._get_observation(lidar_measurements, imu_measurements)

        # Reward
        self.coverage_grid.visited(self.agent.get_sweepers()[0])
        self.coverage_grid.visited(self.agent.get_sweepers()[1])
        reward = self._calculate_reward(action)
        terminated = self._check_collision()
        info: dict[str, Any] = {}

        if self.render_mode == "human":
            self._render_frame(lidar_measurements)

        return obs_flat, reward, terminated, False, info


    def reset(self, *, seed: Optional[int]=None, options: Optional[dict[str, Any]]=None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        self.agent.position = self.environment.next_start_position()
        self.agent.angle = random.uniform(0, 2 * np.pi)
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        lidar_measurements = self.lidar.measurement(self.agent)
        imu_measurements = self.Imu.measurement(self.agent, self.dt)

        return self._get_observation(lidar_measurements, imu_measurements), {}


    def select_environment(self, idx: int) -> None:
        self.environment = self.environments[idx - 1]
        self.lidar = Lidar(self.environment, max_distance=self.max_distance, num_rays=self.num_rays, uncertainty=(1.0, 0.01))


    def get_environment_count(self) -> int:
        return len(self.environments)


    def get_coverage(self) -> int:
        return self.coverage_grid.coverage()


    def get_coverage_percentage(self) -> float:
        return self.coverage_grid.coverage_percentage()


    def _load_environments(self, base_folder: str) -> list[LidarEnvironment]:
        environments: list[LidarEnvironment] = []
        maps_folder = Path(base_folder) / "maps"
        start_positions_folder = Path(base_folder) / "start_positions"

        for map_file in tqdm(sorted(maps_folder.glob("map_*.png")), desc="Loading maps"):
            map_filename = map_file.stem
            start_positions_path = start_positions_folder / f"{map_filename}.npy"

            env = LidarEnvironment(str(map_file), str(start_positions_path), self.map_dimensions)
            environments.append(env)

        return environments


    def _get_observation(self, lidar_measurements: NDArray[np.float32], imu_measurements: NDArray[np.float32]) -> NDArray[np.float32]:
        # measurements shape: (num_rays, 5) -> [distance, angle, hit, x, y]
        distances = np.clip(lidar_measurements[:, 0], 0, self.max_distance) / self.max_distance
        hits: NDArray[np.float32] = lidar_measurements[:, 2]
        obs_2d_normalized = np.stack((distances, hits), axis=1)

        # Normalize accelerations
        lin_acc_normalized: NDArray[np.float32] = np.clip(imu_measurements[:2], -self.v_max, self.v_max) / self.v_max
        ang_acc_normalized: NDArray[np.float32] = np.clip(imu_measurements[2], -self.omega_max, self.omega_max) / self.omega_max
        acceleration_normalized = np.concatenate((lin_acc_normalized, np.array([ang_acc_normalized], dtype=np.float32)), axis=0)

        return np.concatenate((obs_2d_normalized.ravel(), acceleration_normalized), axis=0)


    def _calculate_reward(self, action: NDArray[np.float32]) -> float:
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


    def _render_frame(self, measurements: NDArray[np.float32]) -> None:
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


    def render(self, mode: str='human') -> str:
        return mode


    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



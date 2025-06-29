import time
from typing import Optional, Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
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
        self.a_max = 2000.0

        # Environments setup
        self.map_dimensions = (1200, 600)
        self.environments = self._load_environments("project/generated")

        # Agent setup
        self.lidar: Lidar
        self.Imu: Imu = Imu()
        self.environment: LidarEnvironment
        self.select_environment(1)

        start_position = self.environment.next_start_position()
        self.agent = Agent(position=start_position, angle=0.0)

        # Action and observation space
        self.action_space = spaces.Box(
            low=np.array([-500.0, -1000.0], dtype=np.float32),
            high=np.array([250.0, 1000.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_rays * 2 + 3,), dtype=np.float32
        )

        self.v_max = 250.0
        self.v_min = -50.0
        self.omega_max = 5.0

        # Coverage grid
        self.grid_resolution = 10
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        self.prev_coverage = 0

        # Rewards
        self.reward_coefficients = np.array([
            -0.005 / self.dt,  # time
            -0.25 / self.omega_max,  # omega
            -1000.0,  # collision
            1.0 / self.v_max,  # velocity
            50.0,  # coverage
        ], dtype=np.float32)

        self.step_count = 0
        self.start = time.perf_counter()


    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        
        # Apply unicycle kinematics
        action = np.clip(action, self.action_space.low, self.action_space.high, dtype=np.float32)
        self.agent.apply_action(action, self.dt)

        # LIDAR observation
        lidar_measurements = self.lidar.measurement(self.agent)
        imu_measurements = self.Imu.measurement(self.agent, self.dt)
        obs_flat = self._get_observation(lidar_measurements, imu_measurements)

        # Reward
        self.coverage_grid.mark_visited(self.agent.get_sweepers()[0])
        self.coverage_grid.mark_visited(self.agent.get_sweepers()[1])
        reward_components = self._calculate_reward_components()
        reward = np.sum(reward_components)
        terminated = self._check_collision()

        if self.render_mode == "human":
            self._render_frame(lidar_measurements)

        return obs_flat, reward, terminated, False, self._generate_info(action, obs_flat, reward, reward_components, lidar_measurements, imu_measurements)


    def reset(self, *, seed: Optional[int]=None, options: Optional[dict[str, Any]]=None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_coverage = 0
        self.start = time.perf_counter()

        self.Imu.reset()
        self.agent.reset()
        self.agent.position = self.environment.next_start_position()
        self.coverage_grid = CoverageGridDTO(self.map_dimensions, self.grid_resolution)
        lidar_measurements = self.lidar.measurement(self.agent)
        imu_measurements = self.Imu.measurement(self.agent, self.dt)

        return self._get_observation(lidar_measurements, imu_measurements), {}


    def select_environment(self, idx: int) -> None:
        self.environment = self.environments[idx - 1]
        self.lidar = Lidar(self.environment, max_distance=self.max_distance, num_rays=self.num_rays, uncertainty=(1.0, 0.01))


    def get_environment_count(self) -> int:
        return len(self.environments)


    def _load_environments(self, base_folder: str) -> list[LidarEnvironment]:
        environments: list[LidarEnvironment] = []
        maps_folder = Path(base_folder) / "maps"
        start_positions_folder = Path(base_folder) / "start_positions"

        map_files = sorted(
            maps_folder.glob("map_*.png"),
            key=lambda x: int(x.stem.split('_')[1])
        )

        for map_file in tqdm(map_files, desc="Loading maps"):
            map_filename = map_file.stem
            start_positions_path = start_positions_folder / f"{map_filename}.npy"

            env = LidarEnvironment(map_file, start_positions_path, self.map_dimensions)
            environments.append(env)

        return environments


    def _get_observation(self, lidar_measurements: NDArray[np.float32], imu_measurements: NDArray[np.float32]) -> NDArray[np.float32]:
        # measurements shape: (num_rays, 5) -> [distance, angle, hit, x, y]
        distances = np.clip(lidar_measurements[:, 0], 0, self.max_distance, dtype=np.float32) / self.max_distance
        hits: NDArray[np.float32] = lidar_measurements[:, 2]
        lidar_normalized = np.stack((distances, hits), axis=0)

        imu_normalized: NDArray[np.float32] = np.clip(imu_measurements, -self.a_max, self.a_max, dtype=np.float32) / self.a_max

        return np.concatenate((lidar_normalized.ravel(), imu_normalized), axis=0)


    def _calculate_reward_components(self) -> NDArray[np.float32]:
        current = self.coverage_grid.coverage()
        delta = current - self.prev_coverage
        self.prev_coverage = current
        v, _, omega = self.agent.get_local_velocity()
        
        features = np.array([
            1.0,  # time
            abs(omega),  # omega
            1.0 if self._check_collision() else 0.0,  # collision
            v,  # velocity
            delta,  # coverage
        ], dtype=np.float32)
        
        return features * self.reward_coefficients


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
        
        
    def _generate_info(self, 
                    action: NDArray[np.float32],
                    observation: NDArray[np.float32],
                    reward: float,
                    reward_components: NDArray[np.float32],
                    lidar_measurements: NDArray[np.float32],
                    imu_measurements: NDArray[np.float32]) -> dict[str, Any]:
        return {    
            "step_count":  self.step_count,
            "elapsed_time": time.perf_counter() - self.start,
            "action": action,
            "observation": observation,
            "reward": reward,
            "reward_components": reward_components,
            "coverage": self.coverage_grid.coverage(),
            "coverage_percentage": self.coverage_grid.coverage_percentage(),
            "agent_pose": self.agent.get_pose(),
            "agent_local_velocity": self.agent.get_local_velocity(),
            "lidar_measurements": lidar_measurements,
            "imu_measurements": imu_measurements,
            "environment_name": self.environment.name,
        }


    def render(self, mode: str='human') -> str:
        return mode


    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



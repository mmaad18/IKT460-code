from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium.core import RenderFrame, ActType, ObsType

import Box2D
from Box2D.b2 import vec2


class UniCycleBasicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 512
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # Initialize Box2D world and create agents
        self.world = Box2D.b2World()
        self.robot = self.world.CreateDynamicBody(position=(0, 0))
        self.robot.CreateCircleFixture(radius=0.5, density=1.0, friction=0.3)

        self.window = None
        self.clock = None

    def step(self, action):
        action = action.astype(np.float64)
        force_vector = vec2(*action)
        self.robot.ApplyForceToCenter(force_vector, True)
        self.world.Step(1.0/60.0, 6*30, 2*30)

        observation = self._get_observation()
        reward = self._calculate_reward(observation)
        terminated = self._is_terminated(observation)
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state
        self.robot.position = (0, 0)
        self.robot.linearVelocity = (0, 0)

        info = {}

        return self._get_observation(), info

    def _get_observation(self):
        # Simulate LIDAR readings or other sensor data
        # Return as a numpy array
        return np.random.rand(10).astype(np.float32)

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
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



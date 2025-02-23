import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import world, polygonShape, circleShape, staticBody, dynamicBody

class LidarRobotEnv(gym.Env):
    def __init__(self):
        super(LidarRobotEnv, self).__init__()

        # Box2D world setup
        self.world = world(gravity=(0, 0), doSleep=True)  # No gravity for simplicity
        self.robot = None
        self.obstacles = []

        # Action space: [linear_velocity, angular_velocity] for unicycle
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Observation space: [LIDAR readings, robot position, robot orientation]
        self.num_lidar_rays = 16  # Number of LIDAR rays
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (self.num_lidar_rays + 3)),  # LIDAR + x, y, theta
            high=np.array([np.inf] * (self.num_lidar_rays + 3)),
            dtype=np.float32
        )

        # Simulation parameters
        self.time_step = 1.0 / 60.0  # Physics simulation time step
        self.max_steps = 1000  # Maximum steps per episode
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Reset the environment to initial state
        self.world.ClearForces()
        self.robot = self._create_robot()
        self.obstacles = self._create_obstacles()
        self.current_step = 0

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Apply the action to the robot
        self._apply_action(action)
        self.world.Step(self.time_step, 6, 2)  # Step the physics simulation
        self.current_step += 1

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._compute_reward()

        # Check if the episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False  # Use this if you want to truncate episodes early

        return observation, reward, terminated, truncated

    def render(self, mode='human'):
        # Optional: Render the environment using Pygame or other tools
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # Helper methods
    def _create_robot(self):
        # Create the robot body in Box2D
        return self.world.CreateDynamicBody(
            position=(0, 0),
            fixtures=circleShape(radius=0.5, density=1)
        )

    def _create_obstacles(self):
        # Create static obstacles in the environment
        obstacles = []
        obstacles.append(self.world.CreateStaticBody(
            position=(5, 0),
            fixtures=polygonShape(box=(1, 1))
        ))
        return obstacles

    def _apply_action(self, action):
        # Apply the action (linear_velocity, angular_velocity) to the robot
        linear_vel, angular_vel = action
        self.robot.linearVelocity = (linear_vel, 0)
        self.robot.angularVelocity = angular_vel

    def _get_observation(self):
        # Generate LIDAR readings and combine with robot state
        lidar_readings = self._simulate_lidar()
        robot_position = self.robot.position
        robot_angle = self.robot.angle
        return np.concatenate((lidar_readings, [robot_position.x, robot_position.y, robot_angle]))

    def _simulate_lidar(self):
        # Simulate LIDAR by raycasting in multiple directions
        lidar_readings = []
        for angle in np.linspace(0, 2 * np.pi, self.num_lidar_rays, endpoint=False):
            callback = Box2D.b2RayCastCallback()
            self.world.RayCast(callback, self.robot.position, self.robot.position + Box2D.b2Vec2(np.cos(angle), np.sin(angle)) * 10)
            lidar_readings.append(callback.fraction * 10)  # Distance scaled by max range
        return lidar_readings

    def _compute_reward(self):
        # Example reward function: Penalize collisions, encourage exploration
        reward = 0
        if self._check_collision():
            reward -= 10  # Collision penalty
        else:
            reward += 1  # Small reward for surviving
        return reward

    def _check_collision(self):
        # Check if the robot has collided with any obstacle
        for contact_edge in self.robot.contacts:
            if contact_edge.contact.touching:
                return True
        return False
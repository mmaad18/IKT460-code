import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from unicycle_env.envs.Agent import Agent  # pyright: ignore [reportMissingTypeStubs]


@dataclass
class Imu:
    last_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def measurement(self, agent: Agent, dt: float) -> NDArray[np.float32]:
        pose = agent.get_pose()
        x, y, theta = pose

        # Velocity
        vx = (x - self.last_pose[0]) / dt
        vy = (y - self.last_pose[1]) / dt
        vtheta = (theta - self.last_pose[2]) / dt

        # Acceleration
        ax = (vx - self.last_velocity[0]) / dt
        ay = (vy - self.last_velocity[1]) / dt
        az = (vtheta - self.last_velocity[2]) / dt

        self.last_pose = pose
        self.last_velocity = (vx, vy, vtheta)
        
        print(f"IMU Measurement: pose={pose}, velocity=({vx}, {vy}, {vtheta}), acceleration=({ax}, {ay}, {az})")

        return np.array([ax, ay, az], dtype=np.float32)

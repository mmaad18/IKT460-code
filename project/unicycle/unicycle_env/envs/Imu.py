import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
from unicycle_env.envs.Agent import Agent  # pyright: ignore [reportMissingTypeStubs]


@dataclass
class Imu:
    last_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def measurement(self, agent: Agent, dt: float) -> NDArray[np.float32]:
        vx_local, vy_local, omega_local = agent.get_local_velocity()
        
        ax_local = (vx_local - self.last_velocity[0]) / dt
        ay_local = vx_local * omega_local
        az_local = (omega_local - self.last_velocity[2]) / dt

        self.last_velocity = (vx_local, vy_local, omega_local)

        return np.array([ax_local, ay_local, az_local], dtype=np.float32)

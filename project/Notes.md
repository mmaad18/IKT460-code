# Notes

## About the setup

- Pre generated maps and starting poses.
- Unicyle model with LIDAR.
- 360 degree LIDAR with 60 rays.
- Output:
- 60 rays, 0 - 200 px, normalized to 0 - 1.
- 60 rays hit/no hit, 0.0 or 1.0.
- x, y, theta, normalized to 0 - 1.

## Monte Carlo Control

Implementing Monte Carlo Control for the 2D Unicycle Model seems infeasible due to the continuous state and action spaces. 
Monte Carlo methods typically require discretization of the state and action spaces, which may not be practical for this project. 
Instead, we can focus on implementing Q-Learning or SARSA, which are more suitable for continuous spaces.

## DQN

- Target network is updated every step using soft update.
- 




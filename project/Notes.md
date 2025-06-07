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
 
## A2C

- Source: https://openai.com/index/openai-baselines-acktr-a2c/ 

## TODO 

1) Better logging and plotting.


- Dont use all maps for training
- Maps must be diverse
- Add dynamic obstacles? Too much work?
- Use all maps for training and testing?
- Do acceleration control instead of velocity?
- Save video of training and testing?
- Punish visiting the same place.
- Larger penalty for hitting walls.
- Use CNN for the LIDAR data?
- Use LSTM for the LIDAR data?
- Rank map difficulty by spawning area. More difficult => less area.
- 




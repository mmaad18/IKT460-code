# Title

Autonomous Exploration and Coverage Using LIDAR and Deep Reinforcement Learning

## Project Description

1) 2D Unicycle Model Simulation: Develop a 2D simulation of a LIDAR-equipped unicycle robot for area coverage.
2) RL Algorithm Implementation (2D Unicycle): Implement and compare at least two RL algorithms for the 2D unicycle model.
3) Dynamic Environment Simulation: Introduce moving obstacles. Update RL algorithms to handle dynamic environment.
4) Model Extension (2D Bicycle): Extend the 2D simulation and RL algorithms to bicycle model.

## Simulation Environment Proposals

- Gymnasium + Box2D

### Gymnasium Environments 

- https://gymnasium.farama.org/environments/box2d/car_racing/

## RL Framework Proposals

- Stable-Baselines3 

## Algorithm proposals

- MC (Monte Carlo)
- SARSA (State-Action-Reward-State-Action)
- Q-Learning

- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)

- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)

- SAC (Soft Actor-Critic)
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous Actor-Critic Agents)

- Rainbow DQN? 

## Notes:

- Dont use all maps for training 
- Maps must be diverse
- Add dynamic obstacles? Too much work?
- Use all maps for training and testing?
- Do acceleration control instead of velocity?
- 



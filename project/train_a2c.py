import numpy as np
from tqdm import tqdm

import unicycle_env

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim

from project.rl_algorithms.A2C import A2C
from project.rl_algorithms.DQN import DQN
from project.rl_algorithms.ReplayMemory import ReplayMemory, Transition
from unicycle_env.wrappers import DiscreteActions



action_mapping = [
    [250.0, 0.0],  # Forward
    [-50.0, 0.0],  # Backward
    [0.0, 5.0],  # Turn right
    [0.0, -5.0],  # Turn left
    [250.0, 5.0],  # Forward right
    [250.0, -5.0],  # Forward left
    [-50.0, 5.0],  # Backward right
    [-50.0, -5.0],  # Backward left
]


# environment hyperparams
n_envs = 10
n_updates = 1000
n_steps_per_update = 128
randomize_domain = False

# agent hyperparams
gamma = 0.999
lam = 0.95  # hyperparameter for GAE
ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
actor_lr = 0.001
critic_lr = 0.005

# Note: the actor has a slower learning rate so that the value targets become
# more stationary and are therefore easier to estimate for the critic

# environment setup
envs = gym.vector.SyncVectorEnv([
    lambda: DiscreteActions(gym.make("unicycle_env/UniCycleBasicEnv-v0", render_mode="rgb_array"), action_mapping)
    for _ in range(n_envs)
])

obs_shape = envs.single_observation_space.shape[0]
action_shape = 8

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init the agent
agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)


# create a wrapper environment to save episode returns and episode lengths
#envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs)

critic_losses = []
actor_losses = []
entropies = []

# use tqdm to get a progress bar for training
for sample_phase in tqdm(range(n_updates)):
    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically

    # reset lists that collect experiences of an episode (sample phase)
    ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
    masks = torch.zeros(n_steps_per_update, n_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        states, info = envs.reset(seed=42)

    # play n steps in our parallel environments to collect data
    for step in range(n_steps_per_update):
        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(states)

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())

        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = torch.tensor(rewards, device=device)
        ep_action_log_probs[step] = action_log_probs

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
        masks[step] = torch.tensor([not term for term in terminated])

    # calculate the losses for actor and critic
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        entropy,
        masks,
        gamma,
        lam,
        ent_coef,
        device,
    )

    # update the actor and critic networks
    agent.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    critic_losses.append(critic_loss.detach().cpu().numpy())
    actor_losses.append(actor_loss.detach().cpu().numpy())
    entropies.append(entropy.detach().mean().cpu().numpy())


torch.save(agent.actor.state_dict(), "a2c_actor.pth")
torch.save(agent.critic.state_dict(), "a2c_critic.pth")
envs.close()
print("Training complete.")


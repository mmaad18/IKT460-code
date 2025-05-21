import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from project.rl_algorithms.ReplayMemory import Transition, ReplayMemory
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


class DQN(nn.Module):
    def __init__(
            self,
            device: torch.device,
            input_dim: int,
            output_dim: int,
            learning_rate: float,
            memory_capacity: int,
            eps_start: float,
            eps_end: float,
            eps_decay: int,
            batch_size: int,
            gamma: float
    ) -> None:
        super().__init__()
        self.device = device

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        ).to(self.device)

        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.gamma = gamma

        self.memory = ReplayMemory(self.memory_capacity)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)


    def forward(self, x: np.ndarray) -> torch.Tensor:
        return self.policy_net(x)


    def memory_push(self, transition: Transition) -> None:
        self.memory.push(transition)


    def select_action(self, env: DiscreteActions, state: torch.Tensor, step_count: int) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * step_count / self.eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)


    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Batch-array of Transitions => Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()





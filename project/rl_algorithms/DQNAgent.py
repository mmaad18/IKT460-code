import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from project.rl_algorithms.DQN import DQN
from project.rl_algorithms.ReplayBuffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor  # discount factor
        self.epsilon = epsilon_start  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-Network
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.action_size)
    
    def step(self, state, action, reward, next_state, done):
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every few steps if enough samples are available
        self.t_step = (self.t_step + 1) % self.target_update_freq
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            
            # Update target network
            if self.t_step == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for next states from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Get expected Q values from policy model
        Q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
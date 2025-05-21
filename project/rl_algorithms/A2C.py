import numpy as np
import torch
import torch.nn as nn
from torch import optim

class A2C(nn.Module):
    def __init__(
            self,
            n_features: int,
            n_actions: int,
            device: torch.device,
            critic_lr: float,
            actor_lr: float,
            n_envs: int,
    ) -> None:
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        # Estimate V(s)
        self.critic = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

        # Estimate action logits (will be fed into a softmax later)
        self.actor = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        ).to(self.device)

        # Optimizers for actor and critic
        self.critic_optim = optim.RMSprop(self.critic.parameters(), lr=critic_lr)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=actor_lr)


    def forward(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(x).to(self.device)
        state_values = self.critic(x)  # shape: [n_envs,]
        action_logits_vec = self.actor(x)  # shape: [n_envs, n_actions]
        return (state_values, action_logits_vec)


    def select_action(self, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_values, action_logits = self.forward(x)

        # implicitly uses softmax
        action_pd = torch.distributions.Categorical(logits=action_logits)

        actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        return (actions, action_log_probs, state_values, entropy)


    def get_losses(
            self,
            rewards: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_preds: torch.Tensor,
            entropy: torch.Tensor,
            masks: torch.Tensor,
            gamma: float,
            lam: float,
            ent_coef: float,
            device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()

        return (critic_loss, actor_loss)


    def update_parameters(self, critic_loss: torch.Tensor, actor_loss: torch.Tensor) -> None:
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


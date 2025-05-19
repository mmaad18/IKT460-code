import torch
import torch.nn as nn

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
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

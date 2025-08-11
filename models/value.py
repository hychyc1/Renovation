import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, cfg):
        """
        Value Network for estimating the state value function.

        Args:
        - feature_dim (int): Dimension of the input state encoding.
        """
        super().__init__()
        # Fully connected layers for value estimation
        self.fc1 = nn.Linear(cfg.global_feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output for state value

    def forward(self, x):
        """
        Forward pass through the value network.

        Args:
        - x (torch.Tensor): Encoded state, shape [batch_size, feature_dim].

        Returns:
        - value (torch.Tensor): Estimated state value, shape [batch_size, 1].
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)  # No activation on the output
        return value

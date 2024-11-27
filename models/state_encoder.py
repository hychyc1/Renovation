import torch
import torch.nn as nn
import torch.nn.functional as F
from Config.config import Config

class CNNStateEncoder(nn.Module):
    def __init__(self, cfg: Config):
        """
        Initializes the CNN State Encoder.

        Args:
        - cfg (Config): Configuration object.
        """
        super(CNNStateEncoder, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(len(cfg.grid_attributes), 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: 55x54 -> 27x27x16

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: 27x27 -> 13x13x32

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: 13x13 -> 6x6x64

        # Fully connected layer for final encoding
        # Concatenate features from all levels
        flattened_size = (27 * 27 * 16) + (13 * 13 * 32) + (6 * 6 * 64)
        self.fc = nn.Linear(flattened_size, cfg.feature_dim)

    def forward(self, x):
        """
        Forward pass through the CNN state encoder.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Encoded state with shape (batch_size, feature_dim).
        """
        # First level
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))  # Batch: (batch_size, 16, 55, 54)
        x1_pooled = self.pool1(x1)  # Batch: (batch_size, 16, 27, 27)

        # Second level
        x2 = F.leaky_relu(self.bn2(self.conv2(x1_pooled)))  # Batch: (batch_size, 32, 27, 27)
        x2_pooled = self.pool2(x2)  # Batch: (batch_size, 32, 13, 13)

        # Third level
        x3 = F.leaky_relu(self.bn3(self.conv3(x2_pooled)))  # Batch: (batch_size, 64, 13, 13)
        x3_pooled = self.pool3(x3)  # Batch: (batch_size, 64, 6, 6)

        # Flatten all pooled features, preserving batch dimension
        x1_flat = x1_pooled.view(x.size(0), -1)  # Flatten: (batch_size, 27*27*16)
        x2_flat = x2_pooled.view(x.size(0), -1)  # Flatten: (batch_size, 13*13*32)
        x3_flat = x3_pooled.view(x.size(0), -1)  # Flatten: (batch_size, 6*6*64)

        # Concatenate features along the last dimension
        concatenated_features = torch.cat([x1_flat, x2_flat, x3_flat], dim=1)  # (batch_size, flattened_size)

        # Fully connected layer for final encoding
        encoded_state = self.fc(concatenated_features)  # (batch_size, feature_dim)

        return encoded_state

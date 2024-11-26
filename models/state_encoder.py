import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNStateEncoder(nn.Module):
    def __init__(self, cfg):
        """
        Initializes the CNN State Encoder.

        Args:
        - input_channels (int): Number of input channels (k in n * m * k).
        - feature_dim (int): Dimension of the final feature vector.
        """
        super(CNNStateEncoder, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(cfg.state_encode, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Global average pooling for final feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to map features to the desired dimension
        self.fc = nn.Linear(256 + 128 + 64 + 32, cfg.feature_dim)

    def forward(self, x):
        # First level
        x1 = F.leaky_relu(self.bn1(self.conv1(x)))  # Feature map size: n * m * 32
        
        # Second level
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)))  # Feature map size: n * m * 64
        
        # Third level
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)))  # Feature map size: n * m * 128
        
        # Fourth level
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)))  # Feature map size: n * m * 256
        
        # Global average pooling
        g1 = self.global_pool(x1).view(x.size(0), -1)  # Flatten: batch_size * 32
        g2 = self.global_pool(x2).view(x.size(0), -1)  # Flatten: batch_size * 64
        g3 = self.global_pool(x3).view(x.size(0), -1)  # Flatten: batch_size * 128
        g4 = self.global_pool(x4).view(x.size(0), -1)  # Flatten: batch_size * 256
        
        # Concatenate features from all levels
        concatenated_features = torch.cat([g1, g2, g3, g4], dim=1)
        
        # Fully connected layer for final encoding
        encoded_state = self.fc(concatenated_features)
        
        return encoded_state

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

class FeaturePyramidEncoder(nn.Module):
    """
    CNN-based feature encoder that:
      1) Processes an input grid [B, in_channels, H, W] via multi-scale CNN + FPN.
      2) Computes a global feature by pooling the fused grid features and fusing with an embedded 'year'.
      3) For each village (given by [row, col, area]), it gathers the corresponding grid cell feature,
         concatenates an embedded year feature, processes through an MLP, and explicitly appends the raw area.
    
    The final per-village embedding has its last dimension as the village area.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        # village_feature_dim is the final dimension, including one slot reserved for area.
        self.village_feature_dim = cfg.village_feature_dim  

        # --- 1) CNN Backbone ---
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(len(cfg.grid_attributes), 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # --- 2) FPN Lateral Projections ---
        self.lateral_conv1 = nn.Conv2d(32, self.hidden_dim, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(64, self.hidden_dim, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(128, self.hidden_dim, kernel_size=1)

        # --- 3) Global Feature Branch ---
        # Embed the scalar year into a vector of size hidden_dim.
        self.year_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU()
        )
        # Fuse pooled grid feature and year embedding.
        self.global_fc = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )

        # --- 4) Village Feature Branch ---
        # For each village, we concatenate:
        #   - The grid cell feature at its location (size: hidden_dim)
        #   - The year embedding (size: hidden_dim)
        # The combined vector (size: 2*hidden_dim) is passed through an MLP that outputs (village_feature_dim - 1)
        # features. Finally, the raw area is appended as the last dimension.
        self.village_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.village_feature_dim - 1)
        )

    def forward(self, grid_input: torch.Tensor, village_data: torch.Tensor, year: torch.Tensor):
        """
        Args:
            grid_input: [B, in_channels, H, W]
            village_data: [B, K, 3] with columns (row, col, area)
            year: [B] or [B, 1] scalar input for each batch element
        Returns:
            village_features: [B, K, village_feature_dim] (last dimension is the raw area)
            global_features: [B, global_feature_dim]
        """
        batch_size, _, height, width = grid_input.shape

        # --- Multi-scale CNN Extraction ---
        feat1 = self.conv_block1(grid_input)   # [B, 32, H/2, W/2]
        feat2 = self.conv_block2(feat1)          # [B, 64, H/4, W/4]
        feat3 = self.conv_block3(feat2)          # [B, 128, H/8, W/8]

        # --- FPN-style Fusion ---
        lat1 = self.lateral_conv1(feat1)         # [B, hidden_dim, H/2, W/2]
        lat2 = self.lateral_conv2(feat2)         # [B, hidden_dim, H/4, W/4]
        lat3 = self.lateral_conv3(feat3)         # [B, hidden_dim, H/8, W/8]
        up3 = F.interpolate(lat3, size=lat2.shape[-2:], mode='nearest')
        fuse2 = lat2 + up3
        up2 = F.interpolate(fuse2, size=lat1.shape[-2:], mode='nearest')
        fuse1 = lat1 + up2
        fused_feature_map = F.interpolate(fuse1, size=(height, width), mode='nearest')  # [B, hidden_dim, H, W]

        # --- Global Feature Extraction ---
        global_pool = F.adaptive_avg_pool2d(fused_feature_map, (1, 1)).view(batch_size, -1)  # [B, hidden_dim]
        if year.dim() == 1:
            year = year.unsqueeze(-1)
        year_global = self.year_embed(year.float())  # [B, hidden_dim]
        global_concat = torch.cat([global_pool, year_global], dim=-1)  # [B, 2*hidden_dim]
        global_features = self.global_fc(global_concat)  # [B, global_feature_dim]

        # --- Per-Village Feature Extraction ---
        # village_data: [B, K, 3] where columns = (row, col, area)
        village_coords = village_data[..., :2].long()   # [B, K, 2]
        village_area   = village_data[..., 2].unsqueeze(-1)  # [B, K, 1]
        
        # Gather the grid cell feature for each village
        batch_indices = torch.arange(batch_size, device=grid_input.device).unsqueeze(-1).expand(batch_size, village_coords.size(1))
        loc_feat = fused_feature_map[batch_indices, :, village_coords[..., 0], village_coords[..., 1]]  # [B, K, hidden_dim]
        
        # Embed year for the village branch (broadcast per village)
        year_village = self.year_embed(year.float())  # [B, hidden_dim]
        year_village = year_village.unsqueeze(1).expand(-1, village_coords.size(1), -1)  # [B, K, hidden_dim]
        
        # Concatenate location feature and year embedding
        village_input = torch.cat([loc_feat, year_village], dim=-1)  # [B, K, 2*hidden_dim]
        village_feat_intermediate = self.village_mlp(village_input)  # [B, K, village_feature_dim - 1]
        
        # Append raw area as the last dimension
        village_features = torch.cat([village_feat_intermediate, village_area], dim=-1)  # [B, K, village_feature_dim]

        return village_features, global_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config.config import Config

class FeaturePyramidEncoder(nn.Module):
    """
    A feature-pyramid-like encoder that:
      1) Takes an input grid of shape [B, in_channels, H, W].
      2) Builds a fused 2D feature map [B, hidden_dim, H, W] via multi-scale CNN + FPN logic.
      3) For each village (row, col, area), gathers the cell-level feature and
         concatenates the 'area' scalar, producing per-village embeddings of shape [B, K, out_dim].
      4) Also generates a single global feature per batch element [B, global_out_dim],
         obtained by pooling the fused feature map, and optionally passing it through a small FC.
      5) The final value network (not shown) can use the global features as input.
    """

    def __init__(
        self,
        cfg: Config
    ):
        """
        Args:
            in_channels: Number of channels in the input grid (e.g. base features + area-sum channel).
            hidden_dim: Number of feature channels in the intermediate/fused feature map.
            out_dim: Embedding dimension for per-village features.
            global_out_dim: Dimension of the global feature output (after optional FC).
            use_global_fc: If True, apply a small FC to the pooled global feature before returning it.
        """
        super().__init__()

        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        self.village_feature_dim = cfg.village_feature_dim
        # -----------------------------
        # 1) CNN backbone
        # -----------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample factor of 2
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample factor of 2
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample factor of 2
        )

        # -----------------------------
        # 2) FPN lateral projections
        # -----------------------------
        self.lateral_conv1 = nn.Conv2d(32, self.hidden_dim, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(64, self.hidden_dim, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(128, self.hidden_dim, kernel_size=1)

        # -----------------------------
        # 3) Per-village MLP
        #    We add +1 dimension for the 'area' scalar
        # -----------------------------
        self.village_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, self.village_feature_dim)
        )

        # -----------------------------
        # 4) Optional global FC
        # -----------------------------
        self.global_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )

        # Store for clarity

    def forward(self, grid_input: torch.Tensor, village_data: torch.Tensor):
        """
        Args:
            grid_input: A batch of grid observations [B, in_channels, H, W].
            village_data: A batch of village info [B, K, 3], where each entry = (row, col, area).

        Returns:
            village_features: Per-village embeddings, shape [B, K, out_dim].
            global_features:  Global feature vector, shape [B, global_out_dim].
        """
        batch_size, _, height, width = grid_input.shape

        # -----------------------------
        # 1) Multi-scale CNN extraction
        # -----------------------------
        features_level1 = self.conv_block1(grid_input)  # [B, 32,  H//2,  W//2 ]
        features_level2 = self.conv_block2(features_level1)  # [B, 64,  H//4,  W//4 ]
        features_level3 = self.conv_block3(features_level2)  # [B, 128, H//8,  W//8 ]

        # -----------------------------
        # 2) FPN-style fusion
        # -----------------------------
        lateral_map3 = self.lateral_conv3(features_level3)  # [B, hidden_dim, H//8, W//8]
        lateral_map2 = self.lateral_conv2(features_level2)  # [B, hidden_dim, H//4, W//4]
        lateral_map1 = self.lateral_conv1(features_level1)  # [B, hidden_dim, H//2, W//2]

        # Upsample from level3 to level2
        upsampled_map3 = F.interpolate(lateral_map3, size=lateral_map2.shape[-2:], mode="nearest")
        fused_map2 = lateral_map2 + upsampled_map3

        # Upsample from level2 to level1
        upsampled_map2 = F.interpolate(fused_map2, size=lateral_map1.shape[-2:], mode="nearest")
        fused_map1 = lateral_map1 + upsampled_map2

        # Finally, upsample fused_map1 to the original resolution (H, W)
        fused_feature_map = F.interpolate(fused_map1, size=(height, width), mode="nearest")
        # fused_feature_map => [B, hidden_dim, H, W]

        # -----------------------------
        # 3) Global feature extraction (pooling)
        # -----------------------------
        # Global average pooling => [B, hidden_dim, 1, 1]
        global_pooled = F.adaptive_avg_pool2d(fused_feature_map, (1, 1))
        # Flatten => [B, hidden_dim]
        global_feature = global_pooled.view(batch_size, -1)

        # If desired, pass through an optional FC
        if self.use_global_fc:
            global_feature = self.global_fc(global_feature)  # [B, global_out_dim]

        # -----------------------------
        # 4) Per-village feature extraction
        # -----------------------------
        # village_data => [B, K, 3], columns => (row, col, area)
        village_coords = village_data[..., :2].long()      # [B, K, 2]
        village_areas  = village_data[..., 2].unsqueeze(-1)  # [B, K, 1]

        # Create batch indices [B, K]
        batch_indices = torch.arange(batch_size, device=grid_input.device).unsqueeze(-1)
        batch_indices = batch_indices.expand(-1, village_coords.size(1))  # [B, K]

        # Gather the cell feature from fused_feature_map
        # fused_feature_map => [B, hidden_dim, H, W]
        # => per_village_features => [B, K, hidden_dim]
        per_village_features = fused_feature_map[
            batch_indices,  # [B, K]
            :,
            village_coords[..., 0],  # row
            village_coords[..., 1]   # col
        ]

        # Concatenate area => [B, K, hidden_dim+1]
        per_village_features_plus_area = torch.cat([per_village_features, village_areas], dim=-1)

        # Pass through village MLP => [B, K, out_dim]
        village_features = self.village_mlp(per_village_features_plus_area)

        return village_features, global_feature
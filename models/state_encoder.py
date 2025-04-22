import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

class FeaturePyramidEncoder(nn.Module):
    """
    A feature-pyramid-like CNN encoder that:
      1) Takes an input grid of shape [B, in_channels, H, W].
      2) Builds a fused 2D feature map via multi-scale CNN + FPN logic.
      3) For each village (row, col, area), gathers the cell-level feature and concatenates the area
         to produce per-village embeddings.
      4) Generates a global feature by pooling the fused feature map and fusing it with an embedded "year"
         input, then applying a small FC.
    """

    def __init__(self, cfg: Config):
        """
        Args:
            cfg: Config object with attributes:
                - grid_attributes: List of grid channels.
                - encoder_hidden_dim: Hidden dimension for the feature map.
                - global_feature_dim: Output dimension for the global feature.
                - village_feature_dim: Output dimension for per-village features.
        """
        super().__init__()

        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        self.village_feature_dim = cfg.village_feature_dim

        # -----------------------------
        # 1) CNN backbone
        # -----------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(len(cfg.grid_attributes), 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Downsample factor of 2
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Downsample factor of 2
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Downsample factor of 2
        )

        # -----------------------------
        # 2) FPN lateral projections
        # -----------------------------
        self.lateral_conv1 = nn.Conv2d(32, self.hidden_dim, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(64, self.hidden_dim, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(128, self.hidden_dim, kernel_size=1)

        # -----------------------------
        # 3) Global feature extraction & Year embedding
        # -----------------------------
        # Global average pooling outputs a tensor of shape [B, hidden_dim].
        # We now embed the year and concatenate it (making the input to the FC layer of size 2*hidden_dim).
        self.year_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU()
        )
        self.global_fc = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )

        # -----------------------------
        # 4) Per-village MLP
        # -----------------------------
        self.village_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, self.village_feature_dim)
        )

    def forward(self, grid_input: torch.Tensor, village_data: torch.Tensor, year: torch.Tensor):
        """
        Args:
            grid_input: A batch of grid observations of shape [B, in_channels, H, W].
            village_data: A tensor with village info of shape [B, K, 3] (columns: (row, col, area)).
            year: Tensor with the year input (shape [B] or [B, 1]).
        
        Returns:
            village_features: Per-village embeddings, shape [B, K, village_feature_dim].
            global_features: Global feature vector, shape [B, global_feature_dim].
        """
        batch_size, _, height, width = grid_input.shape

        # -----------------------------
        # 1) Multi-scale CNN extraction
        # -----------------------------
        features_level1 = self.conv_block1(grid_input)      # [B, 32,  H//2, W//2]
        features_level2 = self.conv_block2(features_level1)   # [B, 64,  H//4, W//4]
        features_level3 = self.conv_block3(features_level2)   # [B, 128, H//8, W//8]

        # -----------------------------
        # 2) FPN-style lateral fusion
        # -----------------------------
        lateral_map3 = self.lateral_conv3(features_level3)    # [B, hidden_dim, H//8, W//8]
        lateral_map2 = self.lateral_conv2(features_level2)     # [B, hidden_dim, H//4, W//4]
        lateral_map1 = self.lateral_conv1(features_level1)     # [B, hidden_dim, H//2, W//2]

        upsampled_map3 = F.interpolate(lateral_map3, size=lateral_map2.shape[-2:], mode="nearest")
        fused_map2 = lateral_map2 + upsampled_map3

        upsampled_map2 = F.interpolate(fused_map2, size=lateral_map1.shape[-2:], mode="nearest")
        fused_map1 = lateral_map1 + upsampled_map2

        fused_feature_map = F.interpolate(fused_map1, size=(height, width), mode="nearest")  # [B, hidden_dim, H, W]

        # -----------------------------
        # 3) Global feature extraction & fusion with year
        # -----------------------------
        global_pooled = F.adaptive_avg_pool2d(fused_feature_map, (1, 1))  # [B, hidden_dim, 1, 1]
        global_pooled = global_pooled.view(batch_size, -1)                # [B, hidden_dim]

        # Ensure year tensor is of shape [B, 1]
        if year.dim() == 1:
            year = year.unsqueeze(-1)
        year_feature = self.year_embed(year.float())                     # [B, hidden_dim]

        # Concatenate the grid's pooled feature with the embedded year.
        global_concat = torch.cat([global_pooled, year_feature], dim=-1)   # [B, 2*hidden_dim]
        global_features = self.global_fc(global_concat)                    # [B, global_feature_dim]

        # -----------------------------
        # 4) Per-village feature extraction
        # -----------------------------
        # village_data: [B, K, 3] with columns (row, col, area)
        village_coords = village_data[..., :2].long()      # [B, K, 2]
        village_areas  = village_data[..., 2].unsqueeze(-1)  # [B, K, 1]

        # Prepare batch indices for gathering the corresponding cell features.
        batch_indices = torch.arange(batch_size, device=grid_input.device).unsqueeze(-1).expand(batch_size, village_coords.size(1))
        
        # Gather the cell feature from the fused feature map.
        # fused_feature_map: [B, hidden_dim, H, W] -> location_features: [B, K, hidden_dim]
        location_features = fused_feature_map[
            batch_indices,
            :,
            village_coords[..., 0],
            village_coords[..., 1]
        ]

        # For concatenation, expand the village area feature to match location_features dimensions.
        village_areas_expanded = village_areas.expand_as(location_features)
        per_village_feature_input = torch.cat([location_features, village_areas_expanded], dim=-1)

        # Pass through the village MLP => [B, K, village_feature_dim]
        village_features = self.village_mlp(per_village_feature_input)

        return village_features, global_features


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        row, col = edge_index  # row: source, col: target
        neigh_messages = self.lin_neigh(x)  # [N, out_channels]
        aggregated = torch.zeros_like(neigh_messages)
        aggregated.index_add_(0, col, neigh_messages[row])
        out = self.lin_self(x) + aggregated
        return self.relu(out)

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import Config

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        # edge_index: [2, E] with row: source, col: target
        row, col = edge_index  # each of shape [E]
        neigh_messages = self.lin_neigh(x)  # [N, out_channels]
        # Using index_add_ for aggregation; we assume that every node not receiving messages gets zero.
        aggregated = torch.zeros_like(neigh_messages)
        aggregated.index_add_(0, col, neigh_messages[row])
        out = self.lin_self(x) + aggregated
        return self.relu(out)

class FeatureEncoderGNN(nn.Module):
    """
    GNN-based encoder where both grid cells and villages are nodes.
    - Grid nodes: Nodes corresponding to each cell in the grid.
    - Village nodes: Nodes that have an initial attribute (e.g., area) and connect to their grid cell.
    
    Inputs:
      - grid_input: [B, in_channels, H, W]
      - village_data: [B, K, 3]  (columns: [row, col, area])
      - year: [B] or [B, 1]
      
    Outputs:
      - village_features: [B, K, village_feature_dim]
      - global_features: [B, global_feature_dim]
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        self.village_feature_dim = cfg.village_feature_dim
        self.in_channels = len(cfg.grid_attributes)
        
        # Project raw grid features into latent space.
        self.grid_proj = nn.Linear(self.in_channels, self.hidden_dim)
        # Project village-specific features (here we use the village "area" as the only initial attribute).
        self.village_proj = nn.Linear(1, self.hidden_dim)
        
        # Define a unified set of GNN layers.
        self.gnn_layers = nn.ModuleList([
            GraphConv(self.hidden_dim, self.hidden_dim) for _ in range(3)
        ])
        
        # Embed the year input.
        self.year_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU()
        )
        
        # Global MLP to combine pooled features and year.
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )
        
        # Final MLP for village node outputs.
        self.village_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.village_feature_dim),
            nn.ReLU()
        )
        
        # Cache for the base edge index and its associated grid/village shape.
        self._cached_edge_index = None
        self._cached_shape = None  # (H, W, K)

    def _build_base_edge_index(self, H, W, K, device):
        """
        Build the base edge index for a single instance (one grid and its villages).
        This includes:
          1. Grid-to-grid edges (8-neighbor connectivity).
          2. Village-to-grid edges: each village connects bidirectionally with the grid cell
             at its (row, col).  
             
        Args:
            H, W: Grid height and width.
            K: Number of villages.
            device: Torch device.
            
        Returns:
            base_edge_index: Tensor of shape [2, E_base] for one instance.
        """
        # ---- 1. Grid-to-grid edges ----
        # Build grid edges (8-neighbors)
        grid_edges = []
        for i in range(H):
            for j in range(W):
                src = i * W + j  # grid node index
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            dst = ni * W + nj
                            grid_edges.append((src, dst))
        grid_edges = torch.tensor(grid_edges, dtype=torch.long, device=device).t().contiguous()  # [2, E_grid]
        
        # ---- 2. Village-to-grid edges ----
        # Assume that village_data's spatial relationship is fixed.
        # We use the village (row, col) from the first sample as representative.
        # We assume there are exactly K villages and that each has a unique position.
        # Compute grid index for each village.
        # (We assume that each village connects to exactly one grid cell.)
        # For vectorization, we assume village_coords is given as shape [K, 2] (rows then cols).
        # Here, we simply create:
        #   edge from grid_node[v] -> village_node (indexed as num_grid_nodes + i) and vice versa.
        village_coords = self._fixed_village_coords  # Tensor of shape [K, 2]
        village_grid_indices = village_coords[:, 0] * W + village_coords[:, 1]  # [K]
        # Create two sets of edges (bidirectional)
        village_node_indices = torch.arange(K, device=device) + (H * W)  # village nodes come after grid nodes.
        village_edges_1 = torch.stack([village_grid_indices, village_node_indices], dim=0)
        village_edges_2 = torch.stack([village_node_indices, village_grid_indices], dim=0)
        village_edges = torch.cat([village_edges_1, village_edges_2], dim=1)  # [2, E_village] with E_village = 2*K
        
        # ---- Combine ----
        base_edge_index = torch.cat([grid_edges, village_edges], dim=1)  # [2, E_base]
        return base_edge_index

    def forward(self, grid_input: torch.Tensor, village_data: torch.Tensor, year: torch.Tensor):
        batch_size, _, H, W = grid_input.shape
        num_grid_nodes = H * W
        _, K, _ = village_data.shape  # number of villages
        
        device = grid_input.device
        
        # --- Cache the base edge index once if H, W, K do not change ---
        current_shape = (H, W, K)
        if (self._cached_edge_index is None) or (self._cached_shape != current_shape):
            # For the village edges, assume village coordinates are the same in each sample.
            # Cache the fixed village coordinates from the first sample.
            # They are in village_data[..., :2] and assumed constant across batch.
            self._fixed_village_coords = village_data[0, :, :2].long().to(device)  # shape: [K, 2]
            self._cached_edge_index = self._build_base_edge_index(H, W, K, device)
            self._cached_shape = current_shape

        base_edge_index = self._cached_edge_index  # shape: [2, E_base]
        total_nodes = num_grid_nodes + K  # per instance

        # --- Vectorized batching of the base edge index ---
        # Create an offset for each instance.
        offsets = torch.arange(batch_size, device=device, dtype=base_edge_index.dtype) * total_nodes  # [B]
        # Reshape offsets to broadcast: [1, 1, B]
        offsets = offsets.view(1, 1, batch_size)
        # Expand base_edge_index: [2, E_base] -> [2, E_base, 1]
        base_edge_index_expanded = base_edge_index.unsqueeze(2)
        # Add offsets to get batched edge indices: result shape [2, E_base, B]
        batched_edge_index = base_edge_index_expanded + offsets
        # Reshape to final shape [2, E_total] where E_total = batch_size * E_base
        combined_edge_index = batched_edge_index.reshape(2, -1)

        # --- Process grid and village nodes ---
        # Process grid nodes.
        # grid_input: [B, in_channels, H, W] -> [B, H*W, in_channels]
        grid_features = grid_input.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        grid_features = self.grid_proj(grid_features)  # [B, H*W, hidden_dim]
        
        # Process village nodes (using the village area only).
        # village_data: [B, K, 3] with columns: (row, col, area)
        village_features = self.village_proj(village_data[..., 2].unsqueeze(-1))  # [B, K, hidden_dim]
        
        # Combine grid and village nodes into one list per instance.
        # Ordering: first grid nodes, then village nodes.
        combined_features = torch.cat([grid_features, village_features], dim=1)  # [B, total_nodes, hidden_dim]
        # Flatten to shape [B * total_nodes, hidden_dim] for GNN message passing.
        combined_features = combined_features.view(batch_size * total_nodes, self.hidden_dim)
        
        # --- Run unified GNN layers ---
        for layer in self.gnn_layers:
            combined_features = layer(combined_features, combined_edge_index)
        
        # Reshape back to [B, total_nodes, hidden_dim].
        combined_features = combined_features.view(batch_size, total_nodes, self.hidden_dim)
        
        # --- Global feature extraction ---
        # Use grid nodes only (first num_grid_nodes) and mean pool per instance.
        global_grid_features = combined_features[:, :num_grid_nodes, :].mean(dim=1)  # [B, hidden_dim]
        if year.dim() == 1:
            year = year.unsqueeze(-1)
        year_feature = self.year_mlp(year.float())  # [B, hidden_dim]
        global_features = self.global_mlp(torch.cat([global_grid_features, year_feature], dim=-1))
        
        # --- Village feature extraction ---
        # Extract village nodes (the last K nodes) and pass through village MLP.
        village_nodes_updated = combined_features[:, num_grid_nodes:, :]  # [B, K, hidden_dim]
        village_features = self.village_mlp(village_nodes_updated)  # [B, K, village_feature_dim]
        
        return village_features, global_features


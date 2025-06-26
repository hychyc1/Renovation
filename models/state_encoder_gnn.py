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
        # Using index_add_ for aggregation; nodes not receiving messages get zero.
        aggregated = torch.zeros_like(neigh_messages)
        aggregated.index_add_(0, col, neigh_messages[row])
        out = self.lin_self(x) + aggregated
        return self.relu(out)

class FeatureEncoderGNN(nn.Module):
    """
    GNN-based encoder that incorporates the year into both global and village features.
    
    It builds a single graph (cached across forward passes when grid/village configuration is fixed)
    consisting of grid nodes (each grid cell) and village nodes. The base graph connectivity includes:
      - Grid-to-grid edges (here using 8-neighbor connectivity).
      - Bidirectional edges between each village and its corresponding grid cell.
    
    For the village branch, after GNN processing the village node's updated feature is concatenated
    with the embedded year (via self.year_mlp). The resulting vector (of size 2*hidden_dim) is processed by an MLP,
    whose output (of dimension village_feature_dim - 1) is then concatenated with the raw area as the last dimension.
    
    Inputs:
      - grid_input: [B, in_channels, H, W]
      - village_data: [B, K, 3] (columns: [row, col, area])
      - year: [B] or [B, 1]
      
    Outputs:
      - village_features: [B, K, village_feature_dim]  (last dimension is the raw village area)
      - global_features:  [B, global_feature_dim]
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        # The final village feature dimension reserves one slot for area.
        self.village_feature_dim = cfg.village_feature_dim  
        self.in_channels = len(cfg.grid_attributes)
        
        # Project raw grid cell features.
        self.grid_proj = nn.Linear(self.in_channels, self.hidden_dim)
        # Process village initial attribute (here we use village area as the only input feature) 
        # into the latent space.
        self.village_proj = nn.Linear(1, self.hidden_dim)
        
        # Unified GNN layers.
        self.gnn_layers = nn.ModuleList([
            GraphConv(self.hidden_dim, self.hidden_dim) for _ in range(3)
        ])
        
        # Embed the scalar year (used in both global and village branches).
        self.year_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU()
        )
        
        # Global MLP: fuse mean-pooled grid node features with embedded year.
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )
        
        # Village MLP: takes concatenated [village_node_feature, year_embedding] (2*hidden_dim),
        # outputs a vector of dimension (village_feature_dim - 1) so that raw area can be appended.
        self.village_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.village_feature_dim - 1)
        )
        
        # Cache for the base edge index and associated grid/village shape.
        self._cached_edge_index = None
        self._cached_shape = None  # (H, W, K)
        # Fixed village coordinates (from the first sample) will be cached.
        self._fixed_village_coords = None

    def _build_base_edge_index(self, H, W, K, device):
        """
        Build the base edge index for a single instance (one grid and its villages).
        
        Includes:
          1. Grid-to-grid edges (8-neighbors).
          2. Village-to-grid edges: each village connects bidirectionally with the grid cell at its (row, col).
          
        Args:
            H, W: grid height and width.
            K: number of villages.
            device: torch device.
            
        Returns:
            base_edge_index: Tensor of shape [2, E_base] for one instance.
        """
        # ---- 1. Grid-to-grid edges (8-neighbor connectivity) ----
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
        # Cache fixed village coordinates from the first instance (assumed constant across batch).
        # _fixed_village_coords has shape [K, 2] (rows then cols)
        village_coords = self._fixed_village_coords  # Tensor of shape [K, 2]
        village_grid_indices = village_coords[:, 0] * W + village_coords[:, 1]  # [K]
        # Village nodes come after grid nodes.
        village_node_indices = torch.arange(K, device=device) + (H * W)
        # Create bidirectional edges.
        village_edges_1 = torch.stack([village_grid_indices, village_node_indices], dim=0)
        village_edges_2 = torch.stack([village_node_indices, village_grid_indices], dim=0)
        village_edges = torch.cat([village_edges_1, village_edges_2], dim=1)  # [2, 2*K]
        
        # ---- Combine grid and village edges ----
        base_edge_index = torch.cat([grid_edges, village_edges], dim=1)  # [2, E_base]
        return base_edge_index

    def forward(self, grid_input: torch.Tensor, village_data: torch.Tensor, year: torch.Tensor):
        """
        Args:
            grid_input: [B, in_channels, H, W] (with H and W fixed).
            village_data: [B, K, 3] with columns (row, col, area).
            year: [B] or [B, 1].
            
        Returns:
            village_features: [B, K, village_feature_dim] (last dimension equals raw village area).
            global_features:  [B, global_feature_dim].
        """
        batch_size, _, H, W = grid_input.shape
        num_grid_nodes = H * W
        _, K, _ = village_data.shape
        device = grid_input.device
        
        # --- Cache edge connectivity once if shape changes ---
        current_shape = (H, W, K)
        if (self._cached_edge_index is None) or (self._cached_shape != current_shape):
            # Cache fixed village coordinates from the first sample (rows, cols).
            self._fixed_village_coords = village_data[0, :, :2].long().to(device)  # [K, 2]
            self._cached_edge_index = self._build_base_edge_index(H, W, K, device)
            self._cached_shape = current_shape
        base_edge_index = self._cached_edge_index  # [2, E_base]
        total_nodes = num_grid_nodes + K  # total nodes per instance
        
        # --- Batch the base edge index ---
        # For each instance, offset the node indices.
        offsets = torch.arange(batch_size, device=device, dtype=base_edge_index.dtype) * total_nodes  # [B]
        offsets = offsets.view(1, 1, batch_size)  # shape: [1, 1, B]
        base_edge_index_expanded = base_edge_index.unsqueeze(2)  # [2, E_base, 1]
        batched_edge_index = base_edge_index_expanded + offsets  # [2, E_base, B]
        combined_edge_index = batched_edge_index.view(2, -1)  # [2, batch_size * E_base]
        
        # --- Process grid and village nodes ---
        # Process grid nodes.
        # grid_input: [B, in_channels, H, W] -> [B, H*W, in_channels]
        grid_features = grid_input.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
        grid_features = self.grid_proj(grid_features)  # [B, H*W, hidden_dim]
        
        # Process village nodes using village area.
        # village_data: [B, K, 3] with columns: (row, col, area)
        village_features = self.village_proj(village_data[..., 2].unsqueeze(-1))  # [B, K, hidden_dim]
        
        # Combine grid and village nodes (grid nodes first).
        combined_features = torch.cat([grid_features, village_features], dim=1)  # [B, total_nodes, hidden_dim]
        combined_features = combined_features.view(batch_size * total_nodes, self.hidden_dim)
        
        # --- GNN Message Passing ---
        for layer in self.gnn_layers:
            combined_features = layer(combined_features, combined_edge_index)
        combined_features = combined_features.view(batch_size, total_nodes, self.hidden_dim)
        
        # --- Global Feature Extraction ---
        # Use grid nodes only (first num_grid_nodes), mean pool per instance.
        global_grid_features = combined_features[:, :num_grid_nodes, :].mean(dim=1)  # [B, hidden_dim]
        if year.dim() == 1:
            year = year.unsqueeze(-1)
        year_feature = self.year_mlp(year.float())  # [B, hidden_dim]
        global_features = self.global_mlp(torch.cat([global_grid_features, year_feature], dim=-1))
        
        # --- Village Feature Extraction ---
        # Extract village nodes (last K nodes) from the combined features.
        village_nodes_updated = combined_features[:, num_grid_nodes:, :]  # [B, K, hidden_dim]
        # Embed year for village branch.
        year_village = self.year_mlp(year.float())  # [B, hidden_dim]
        year_village = year_village.unsqueeze(1).expand(-1, K, -1)  # [B, K, hidden_dim]
        # Concatenate updated village node feature with embedded year.
        village_input = torch.cat([village_nodes_updated, year_village], dim=-1)  # [B, K, 2*hidden_dim]
        village_feat_intermediate = self.village_mlp(village_input)  # [B, K, village_feature_dim - 1]
        # Append the raw village area (as the last dimension).
        raw_area = village_data[..., 2].unsqueeze(-1)  # [B, K, 1]
        village_features = torch.cat([village_feat_intermediate, raw_area], dim=-1)  # [B, K, village_feature_dim]
        
        return village_features, global_features

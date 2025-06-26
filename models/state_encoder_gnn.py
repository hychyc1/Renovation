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
        row, col = edge_index
        neigh_messages = self.lin_neigh(x)          # [N, out_channels]
        aggregated = torch.zeros_like(neigh_messages)
        aggregated.index_add_(0, col, neigh_messages[row])
        out = self.lin_self(x) + aggregated
        return self.relu(out)

class FeatureEncoderGNN(nn.Module):
    """
    GNN-based encoder that incorporates year and area_so_far into both global and village features.

    Inputs:
      - grid_input: [B, in_channels, H, W]
      - village_data: [B, K, 3] (columns: [row, col, area])
      - year: [B] or [B,1]
      - area_so_far: [B] or [B,1]

    Outputs:
      - village_features: [B, K, village_feature_dim]
      - global_features:  [B, global_feature_dim]
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.hidden_dim = cfg.encoder_hidden_dim
        self.global_feature_dim = cfg.global_feature_dim
        self.village_feature_dim = cfg.village_feature_dim
        self.in_channels = len(cfg.grid_attributes)

        # Project raw grid & village area
        self.grid_proj    = nn.Linear(self.in_channels, self.hidden_dim)
        self.village_proj = nn.Linear(1, self.hidden_dim)

        # GNN layers
        self.gnn_layers   = nn.ModuleList([
            GraphConv(self.hidden_dim, self.hidden_dim) for _ in range(3)
        ])

        # Embed year and area_so_far
        self.year_mlp     = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.ReLU())
        self.area_mlp     = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.ReLU())

        # Global MLP: fuse grid, year, area embeddings
        self.global_mlp   = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, self.global_feature_dim),
            nn.ReLU()
        )

        # Village MLP: fuse village node, year, area embeddings
        self.village_mlp  = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.village_feature_dim - 1)
        )

        # Cache for edges
        self._cached_edge_index = None
        self._cached_shape      = None
        self._fixed_village_coords = None

    def _build_base_edge_index(self, H, W, K, device):
        # 1) Grid-to-grid (8 neighbors)
        grid_edges = []
        for i in range(H):
            for j in range(W):
                src = i * W + j
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            dst = ni * W + nj
                            grid_edges.append((src, dst))
        grid_edges = torch.tensor(grid_edges, dtype=torch.long, device=device).t().contiguous()

        # 2) Village-to-grid edges
        coords = self._fixed_village_coords  # [K,2]
        grid_idx = coords[:,0] * W + coords[:,1]
        village_idx = torch.arange(K, device=device) + (H * W)
        e1 = torch.stack([grid_idx, village_idx], dim=0)
        e2 = torch.stack([village_idx, grid_idx], dim=0)
        village_edges = torch.cat([e1, e2], dim=1)

        # Combine
        return torch.cat([grid_edges, village_edges], dim=1)

    def forward(self, grid_input, village_data, year, area_so_far):
        B, _, H, W = grid_input.shape
        K = village_data.size(1)
        device = grid_input.device
        # print(grid_input)

        # Cache or build edges
        shape = (H, W, K)
        if self._cached_edge_index is None or self._cached_shape != shape:
            self._fixed_village_coords = village_data[0, :, :2].long().to(device)
            base = self._build_base_edge_index(H, W, K, device)
            self._cached_edge_index = base
            self._cached_shape = shape
        base = self._cached_edge_index

        # Batch edges
        total_nodes = H*W + K
        offsets = torch.arange(B, device=device) * total_nodes  # [B]
        offsets = offsets.view(1, 1, B)
        be = base.unsqueeze(2) + offsets
        batched_edge_index = be.view(2, -1)

        # Project grid
        g = grid_input.view(B, self.in_channels, -1).permute(0,2,1)
        g = self.grid_proj(g)  # [B, H*W, hidden]

        # Project village area
        v = self.village_proj(village_data[...,2].unsqueeze(-1))  # [B, K, hidden]

        # Combine and run GNN
        x = torch.cat([g, v], dim=1).view(B*total_nodes, self.hidden_dim)
        for layer in self.gnn_layers:
            x = layer(x, batched_edge_index)
        x = x.view(B, total_nodes, self.hidden_dim)

        # Global features
        grid_pool = x[:, :H*W, :].mean(dim=1)  # [B, hidden]
        year_in = year.unsqueeze(-1) if year.dim()==1 else year
        area_in = area_so_far.unsqueeze(-1) if area_so_far.dim()==1 else area_so_far
        ye = self.year_mlp(year_in.float())
        ae = self.area_mlp(area_in.float())
        gl_in = torch.cat([grid_pool, ye, ae], dim=-1)
        global_feats = self.global_mlp(gl_in)

        # Village features
        vn = x[:, H*W:, :]  # [B, K, hidden]
        ye_v = ye.unsqueeze(1).expand(-1, K, -1)
        ae_v = ae.unsqueeze(1).expand(-1, K, -1)
        v_in = torch.cat([vn, ye_v, ae_v], dim=-1)
        v_mid = self.village_mlp(v_in)
        raw_area = village_data[...,2].unsqueeze(-1)
        village_feats = torch.cat([v_mid, raw_area], dim=-1)

        return village_feats, global_feats

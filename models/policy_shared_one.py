import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.config import Config

class VillagePolicySharedAltogether(nn.Module):
    """
    Single-stage policy: for each village we produce `num_modes` logits,
    flatten them into a single vector of length N * num_modes, and sample
    exactly one (village, mode) pair.  Assumes village_per_step = 1.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # how many discrete modes per village
        self.num_comb  = len(cfg.combinations)
        self.num_far   = len(cfg.FAR_values)
        self.num_modes = self.num_comb * self.num_far

        # number of villages & embedding dim
        self.N   = cfg.total_villages
        self.dim = cfg.village_feature_dim

        # === Mode head (DeepSets-style) ===
        # per-node encoder
        self.hidden_dims_mode = cfg.hidden_dims_mode
        self.mode_encoder = self.build_mlp(self.dim, self.hidden_dims_mode)
        mode_phi_dim = self.hidden_dims_mode[-1]

        # mode scorer
        combined_dim = mode_phi_dim * 2
        self.mode_mlp  = self.build_mlp(combined_dim, self.hidden_dims_mode)
        self.mode_head = nn.Linear(self.hidden_dims_mode[-1], self.num_modes)

    def build_mlp(self, in_dim, hidden_dims):
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU())
            prev = h
        return nn.Sequential(*layers)

    def scale_logits(self, logits: torch.Tensor, limit=20.0, dim=-1):
        max_abs = torch.max(torch.abs(logits), dim=dim, keepdim=True).values
        scale  = torch.clamp(max_abs / limit, min=1.0)
        return logits / scale

    def forward(self, village_embeddings: torch.Tensor):
        """
        Args:
          village_embeddings: [B, N, dim]

        Returns:
          logits_flat: [B, N * num_modes]  – unnormalized scores over all
                       (village, mode) pairs.
        """
        B, N, D = village_embeddings.size()
        assert N == self.N and D == self.dim

        # 1) per-node encoding
        h = self.mode_encoder(village_embeddings)       # [B,N,phi]
        # 2) global summary
        g = h.mean(dim=1)                               # [B,phi]
        g_exp = g.unsqueeze(1).expand(-1, N, -1)        # [B,N,phi]
        # 3) combine local + global
        h_comb = torch.cat([h, g_exp], dim=-1)          # [B,N,2*phi]
        # 4) MLP + head
        x = self.mode_mlp(h_comb)                       # [B,N,hidden]
        logits = self.mode_head(x)                      # [B,N,num_modes]
        logits = self.scale_logits(logits, dim=-1)

        # 5) mask out villages with zero-area (replicate mask per mode)
        area = village_embeddings[..., -1]               # [B,N]
        mask = (area == 0.0).unsqueeze(-1).expand(-1, -1, self.num_modes)
        logits = logits.masked_fill(mask, -(2**30))

        # 6) flatten
        logits_flat = logits.reshape(B, -1)             # [B, N*num_modes]
        return logits_flat

    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action=False):
        """
        Samples (or picks argmax if mean_action=True) one index in [0, N*num_modes),
        then unpacks it into (village, comb_idx, far_idx).
        """
        logits_flat = self.forward(village_embeddings)  # [B, L]
        B, L = logits_flat.size()

        if mean_action:
            chosen     = logits_flat.argmax(dim=-1)     # [B]
            total_logp = torch.zeros(B, device=logits_flat.device)
            total_ent  = torch.zeros(B, device=logits_flat.device)
        else:
            dist       = Categorical(logits=logits_flat)
            chosen     = dist.sample()                  # [B]
            total_logp = dist.log_prob(chosen)          # [B]
            total_ent  = dist.entropy()                 # [B]

        # unpack indices
        actions = []
        for b, idx in enumerate(chosen.tolist()):
            v = idx // self.num_modes
            m = idx %  self.num_modes
            comb_idx = m // self.num_far
            far_idx  = m %  self.num_far
            actions.append([(v, comb_idx, far_idx)])
        return actions, total_logp, total_ent

    def get_log_prob_entropy(self, village_embeddings: torch.Tensor,
                             batched_actions, mean_action=False):
        """
        Computes log-prob & entropy for provided batched_actions = [[(v,c,f)], ...].
        """
        logits_flat = self.forward(village_embeddings)  # [B, L]
        B, L = logits_flat.size()

        # build flatten indices from (v, c, f)
        idxs = []
        for acts in batched_actions:
            v, c, f = acts[0]
            idxs.append(v * self.num_modes + c * self.num_far + f)
        idxs = torch.tensor(idxs, device=logits_flat.device)

        if mean_action:
            logp = torch.zeros(B, device=logits_flat.device)
            ent  = torch.zeros(B, device=logits_flat.device)
        else:
            dist = Categorical(logits=logits_flat)
            logp = dist.log_prob(idxs)  # [B]
            ent  = dist.entropy()       # [B]
        return logp, ent

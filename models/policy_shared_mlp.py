import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial
from utils.config import Config

class VillagePolicyShared(nn.Module):
    """
    Two-stage policy using DeepSets-style heads:
    1) forward_selection: set-based scoring per village + global context
    2) forward_modes: DeepSets-style mode scorer (per-node encoder + global aggregation)

    Sampling: approximate selection entropy via Multinomial without replacement.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_comb = len(cfg.combinations)
        self.num_far  = len(cfg.FAR_values)
        self.num_modes = self.num_comb * self.num_far
        self.village_per_step = cfg.village_per_step

        # === Selection head (DeepSets) ===
        self.N   = cfg.total_villages
        self.dim = cfg.village_feature_dim

        # 1) per-node encoder phi: dim -> hidden_dims_encoder
        self.hidden_dims_selection = cfg.hidden_dims_selection
        self.encoder = self.build_mlp(self.dim, self.hidden_dims_selection)
        phi_dim = self.hidden_dims_selection[-1]

        # 2) scorer psi: [phi_dim*2] -> hidden_dims_selection -> 1
        combined_dim = phi_dim * 2
        self.selection_mlp = self.build_mlp(combined_dim, self.hidden_dims_selection)
        self.select_head   = nn.Linear(self.hidden_dims_selection[-1], 1)

        # === Mode head (DeepSets) ===
        # 1) per-node encoder for mode: dim -> hidden_dims_mode
        self.hidden_dims_mode = cfg.hidden_dims_mode
        self.mode_encoder = self.build_mlp(self.dim, self.hidden_dims_mode)
        mode_phi_dim = self.hidden_dims_mode[-1]
        # 2) mode scorer: [mode_phi_dim*2] -> hidden_dims_mode -> num_modes
        mode_combined_dim = mode_phi_dim * 2
        self.mode_mlp = self.build_mlp(mode_combined_dim, self.hidden_dims_mode)
        self.mode_head = nn.Linear(self.hidden_dims_mode[-1], self.num_modes)

    def build_mlp(self, in_dim, hidden_dims):
        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.LeakyReLU())
            prev_dim = hd
        return nn.Sequential(*layers)

    def scale_logits(self, logits: torch.Tensor, limit=15.0, dim=-1):
        max_abs = torch.max(torch.abs(logits), dim=dim, keepdim=True).values
        scale_factor = torch.clamp(max_abs / limit, min=1.0)
        return logits / scale_factor

    def forward_selection(self, village_embeddings: torch.Tensor):
        B, N, E = village_embeddings.size()
        assert N == self.N and E == self.dim

        # 1) per-node encode
        h = self.encoder(village_embeddings)            # [B,N,phi_dim]
        # 2) global summary
        g = h.mean(dim=1)                               # [B,phi_dim]
        # 3) concat
        g_exp = g.unsqueeze(1).expand(-1, N, -1)        # [B,N,phi_dim]
        h_comb = torch.cat([h, g_exp], dim=-1)          # [B,N,2*phi_dim]
        # 4) scorer
        x = self.selection_mlp(h_comb)                  # [B,N,hidden_sel]
        logits = self.select_head(x).squeeze(-1)        # [B,N]
        logits = self.scale_logits(logits, dim=-1)
        # 5) mask zero-area
        area_vals = village_embeddings[..., -1]
        logits = logits.masked_fill(area_vals==0.0, -(2**30))
        return logits

    def forward_modes(self, subset_embeddings: torch.Tensor):
        B, k, E = subset_embeddings.size()
        assert E == self.dim

        # 1) per-node encode
        h = self.mode_encoder(subset_embeddings)        # [B,k,mode_phi_dim]
        # 2) global summary
        g = h.mean(dim=1)                               # [B,mode_phi_dim]
        # 3) concat
        g_exp = g.unsqueeze(1).expand(-1, k, -1)        # [B,k,mode_phi_dim]
        h_comb = torch.cat([h, g_exp], dim=-1)          # [B,k,2*mode_phi_dim]
        # 4) mode MLP + head
        x = self.mode_mlp(h_comb)                       # [B,k,hidden_mode]
        logits = self.mode_head(x)                      # [B,k,num_modes]
        logits = self.scale_logits(logits, dim=-1)
        return logits

    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action=False):
        logits_sel = self.forward_selection(village_embeddings)
        B, N = logits_sel.size()

        if mean_action:
            _, chosen = torch.topk(logits_sel, self.village_per_step, dim=-1)
            sel_logp = torch.zeros(B, device=logits_sel.device)
            sel_ent  = torch.zeros(B, device=logits_sel.device)
        else:
            probs = F.softmax(logits_sel, dim=-1)
            chosen = torch.multinomial(probs, self.village_per_step, replacement=False)
            sel_logp = torch.log(probs.gather(1, chosen)).sum(dim=1)
            sel_ent  = Multinomial(self.village_per_step, probs=probs).entropy()

        batch_idx = torch.arange(B, device=chosen.device).unsqueeze(-1)
        subset = village_embeddings[batch_idx, chosen, :]
        logits_mode = self.forward_modes(subset)

        if mean_action:
            modes = torch.argmax(logits_mode, dim=-1)
            dist_m = Categorical(logits=logits_mode)
            mode_logp = dist_m.log_prob(modes).sum(dim=1)
            mode_ent  = dist_m.entropy().sum(dim=1)
        else:
            dist_m = Categorical(logits=logits_mode)
            modes = dist_m.sample()
            mode_logp = dist_m.log_prob(modes).sum(dim=1)
            mode_ent  = dist_m.entropy().sum(dim=1)

        total_logp = sel_logp + mode_logp
        total_ent  = sel_ent + mode_ent

        actions = []
        for b in range(B):
            acts = []
            for i in range(self.village_per_step):
                v = chosen[b, i].item()
                m = modes[b, i].item()
                comb_idx = m // self.num_far
                far_idx  = m % self.num_far
                acts.append((v, comb_idx, far_idx))
            actions.append(acts)
        return actions, total_logp, total_ent

    def get_log_prob_entropy(self, village_embeddings: torch.Tensor, batched_actions, mean_action=False):
        logits_sel = self.forward_selection(village_embeddings)
        B, N = logits_sel.size()

        chosen = torch.tensor([[a[0] for a in acts] for acts in batched_actions], device=logits_sel.device)

        if mean_action:
            sel_logp = torch.zeros(B, device=logits_sel.device)
            sel_ent  = torch.zeros(B, device=logits_sel.device)
        else:
            probs = F.softmax(logits_sel, dim=-1)
            sel_logp = torch.log(probs.gather(1, chosen)).sum(dim=1)
            sel_ent  = Multinomial(self.village_per_step, probs=probs).entropy()

        batch_idx = torch.arange(B, device=chosen.device).unsqueeze(-1)
        subset = village_embeddings[batch_idx, chosen, :]
        logits_mode = self.forward_modes(subset)

        mode_ids = torch.tensor([[c*self.num_far + f for _,c,f in acts]
                                  for acts in batched_actions], device=logits_mode.device)
        dist_m = Categorical(logits=logits_mode)
        mode_logp = dist_m.log_prob(mode_ids).sum(dim=1)
        mode_ent  = dist_m.entropy().sum(dim=1)

        return sel_logp + mode_logp, sel_ent + mode_ent

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial
from utils.config import Config

class VillagePolicy(nn.Module):
    """
    Two-stage policy with flattened MLPs:
      1) forward_selection: [B,N,dim] -> flatten -> MLP -> [B,N]
         -> mask out villages with area=0 by setting selection_logits to -2^30
      2) forward_modes: gather subset => [B,k,dim] -> flatten -> MLP -> [B,k,num_modes]

    During sampling, we approximate selection entropy via Multinomial(...) 
    with replacement, even though we pick villages without replacement.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.num_comb = len(cfg.combinations)
        self.num_far  = len(cfg.FAR_values)
        self.num_modes = self.num_comb * self.num_far
        self.village_per_step = cfg.village_per_step

        # For stage 1 (selection)
        self.N   = cfg.total_villages
        self.dim = cfg.village_feature_dim
        input_sel_dim = self.N * self.dim

        self.hidden_dims_selection = cfg.hidden_dims_selection
        self.selection_mlp = self.build_mlp(input_sel_dim, self.hidden_dims_selection)
        self.select_head   = nn.Linear(self.hidden_dims_selection[-1], self.N)

        # For stage 2 (mode)
        input_mode_dim = self.village_per_step * self.dim
        self.hidden_dims_mode = cfg.hidden_dims_mode
        self.mode_mlp = self.build_mlp(input_mode_dim, self.hidden_dims_mode)
        self.mode_head= nn.Linear(self.hidden_dims_mode[-1], self.village_per_step * self.num_modes)

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

    ###########################################################################
    # forward_selection => produce [B,N], then mask out villages with area=0
    ###########################################################################
    def forward_selection(self, village_embeddings: torch.Tensor):
        """
        Input: [B,N,dim]. Flatten -> MLP -> [B,N]. Then for each village with area=0,
        set selection_logits[b,i] = -2^30.
        The area is the last dimension => village_embeddings[..., -1].
        """
        B, N, E = village_embeddings.shape
        # flatten => [B, N*E]
        x = village_embeddings.view(B, N*E)
        x = self.selection_mlp(x)      # [B, hidden_dims_selection[-1]]
        logits = self.select_head(x)   # [B, N]
        logits = self.scale_logits(logits, dim=-1)  # scale

        # Now mask out villages with area=0
        # area_mask => shape [B,N], True if area=0
        # area is in last dimension => village_embeddings[b,i,-1].
        # We'll do a gather or simply build a boolean mask
        area_values = village_embeddings[..., -1]  # shape [B,N]
        zero_area_mask = (area_values == 0.0)      # shape [B,N], True where area=0

        # set -2^30 => an extremely negative number 
        # do in place or with a masked_fill
        NEG_LARGE = - (2 ** 30)
        logits = logits.masked_fill(zero_area_mask, NEG_LARGE)

        return logits

    ###########################################################################
    # forward_modes => produce [B,k,num_modes]
    ###########################################################################
    def forward_modes(self, subset_embeddings: torch.Tensor):
        """
        subset_embeddings: [B, k, dim]
        => flatten => pass MLP => final => [B, k*num_modes]
        => reshape => [B, k, num_modes]
        """
        B, k, E = subset_embeddings.shape
        x = subset_embeddings.view(B, k*E)
        x = self.mode_mlp(x)
        out = self.mode_head(x)
        out = self.scale_logits(out, dim=-1)
        out = out.view(B, k, self.num_modes)
        return out

    ###########################################################################
    # select_action
    ###########################################################################
    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action=False):
        """
        Return (actions_list, total_log_prob, total_entropy).
        actions_list => [B][k], each => (v_idx, comb_idx, far_idx)
        """
        B, N, E = village_embeddings.shape
        selection_logits = self.forward_selection(village_embeddings)  # [B,N]

        # 1) Pick k villages
        if mean_action:
            # top-k
            _, chosen_indices = torch.topk(selection_logits, k=self.village_per_step, dim=-1)
            # selection log-prob => 0
            selection_log_prob = torch.zeros(B, device=selection_logits.device)
            selection_entropy  = torch.zeros(B, device=selection_logits.device)
        else:
            # sample from softmax
            probs_all = F.softmax(selection_logits, dim=-1)  # [B,N]
            chosen_indices = torch.multinomial(probs_all, self.village_per_step, replacement=False)
            # approximate sum of log p for chosen
            chosen_probs = probs_all.gather(dim=1, index=chosen_indices)  # [B,k]
            selection_log_prob = torch.log(chosen_probs).sum(dim=1)       # [B]

            # *** Approx. selection entropy with a multinomial distribution (with replacement) ***
            dist_sel = Multinomial(self.village_per_step, probs=probs_all) 
            selection_entropy = dist_sel.entropy()  # [B]

        # 2) gather subset => [B,k,dim]
        batch_idx = torch.arange(B, device=chosen_indices.device).unsqueeze(-1)
        subset_emb = village_embeddings[batch_idx, chosen_indices, :]  # [B,k,dim]

        # 3) forward_modes => [B,k,num_modes]
        mode_logits = self.forward_modes(subset_emb)

        if mean_action:
            chosen_modes = torch.argmax(mode_logits, dim=-1)  # [B,k]
            dist = Categorical(logits=mode_logits)
            mode_log_probs = dist.log_prob(chosen_modes)       # [B,k]
            mode_entropy   = dist.entropy()                    # [B,k]
        else:
            dist = Categorical(logits=mode_logits)
            chosen_modes = dist.sample()  # [B,k]
            mode_log_probs = dist.log_prob(chosen_modes)
            mode_entropy   = dist.entropy()

        # sum up logs + ent
        total_lp = selection_log_prob + mode_log_probs.sum(dim=1)
        total_ent= selection_entropy + mode_entropy.sum(dim=1)

        # build final actions
        actions_list = []
        for b in range(B):
            row_actions = []
            row_chosen_indices = chosen_indices[b]  # shape [k]
            row_chosen_modes   = chosen_modes[b]    # shape [k]
            for i in range(self.village_per_step):
                v_idx = row_chosen_indices[i].item()
                m_idx = row_chosen_modes[i].item()
                comb_idx = m_idx // self.num_far
                far_idx  = m_idx % self.num_far
                row_actions.append((v_idx, comb_idx, far_idx))
            actions_list.append(row_actions)

        return actions_list, total_lp, total_ent

    ###########################################################################
    # get_log_prob_entropy => re-run selection => gather => re-run mode => sum
    ###########################################################################
    def get_log_prob_entropy(self, village_embeddings: torch.Tensor, batched_actions, mean_action=False):
        """
        gather all subset embeddings in one shot => forward_modes,
        final loop for clarity with approximate selection log-prob and
        approximate selection entropy via a 'Multinomial' distribution
        for the selection step (with replacement).
        """
        B, N, E = village_embeddings.shape
        selection_logits = self.forward_selection(village_embeddings)  # [B,N]

        # 1) build chosen_indices from batched_actions => shape [B,k]
        chosen_indices_list = []
        for b in range(B):
            row_idx = [ (v_idx) for (v_idx, c_idx, f_idx) in batched_actions[b] ]
            chosen_indices_list.append(row_idx)
        chosen_indices = torch.tensor(chosen_indices_list, device=village_embeddings.device, dtype=torch.long)

        # 2) approximate selection log_prob
        if mean_action:
            selection_log_prob = torch.zeros(B, device=selection_logits.device)
            selection_entropy  = torch.zeros(B, device=selection_logits.device)
        else:
            probs_all = F.softmax(selection_logits, dim=-1)  # [B,N]
            chosen_probs = probs_all.gather(dim=1, index=chosen_indices) # [B,k]
            selection_log_prob = torch.log(chosen_probs).sum(dim=1)      # [B]

            dist_sel = Multinomial(self.village_per_step, probs=probs_all)
            selection_entropy  = dist_sel.entropy()   # [B]

        # 3) gather subset => [B,k,dim]
        b_idx = torch.arange(B, device=chosen_indices.device).unsqueeze(-1)  # [B,1]
        subset_emb = village_embeddings[b_idx, chosen_indices, :]  # [B,k,dim]

        # 4) forward_modes => [B,k,num_modes]
        mode_logits = self.forward_modes(subset_emb)

        # build [B,k] of chosen mode indices
        chosen_mode_ids = []
        for b in range(B):
            row_ids = []
            for (v_idx, c_idx, f_idx) in batched_actions[b]:
                mode_id = c_idx * self.num_far + f_idx
                row_ids.append(mode_id)
            chosen_mode_ids.append(row_ids)
        chosen_mode_ids_t = torch.tensor(chosen_mode_ids, device=mode_logits.device, dtype=torch.long)

        # distribution => shape [B,k]
        dist_mode = Categorical(logits=mode_logits)
        mode_entropy = dist_mode.entropy().sum(dim=1)  # [B]

        # gather mode log prob in a loop for clarity
        total_lp_list = []
        for b in range(B):
            row_lp = 0.0
            row_mode_logits = mode_logits[b]    # [k,num_modes]
            row_mode_ids    = chosen_mode_ids_t[b]  # [k]
            for i in range(self.village_per_step):
                mode_dist_i = Categorical(logits=row_mode_logits[i])
                lp_i = mode_dist_i.log_prob(row_mode_ids[i])
                row_lp += lp_i
            total_lp_list.append(row_lp)

        mode_lp = torch.stack(total_lp_list, dim=0)  # [B]

        total_lp = selection_log_prob + mode_lp
        total_ent= selection_entropy + mode_entropy

        return total_lp, total_ent

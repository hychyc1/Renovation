import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


###############################################################################
# SimpleSelfAttentionBlock
# A multi-head self-attention + feedforward sub-block for global context.
###############################################################################
class SimpleSelfAttentionBlock(nn.Module):
    """
    A single-layer multi-head self-attention + feedforward block.
    """

    def __init__(self, cfg):
        """
        Args:
          cfg: A config object with:
            - cfg.num_attention_heads (int)
            - cfg.feedforward_hidden_dim (int)
            - cfg.attention_dropout (float)
            - cfg.village_feature_dim (int).
        """
        super().__init__()
        self.village_feature_dim = cfg.village_feature_dim
        self.num_attention_heads = cfg.num_attention_heads
        self.feedforward_hidden_dim = cfg.feedforward_hidden_dim
        self.attention_dropout = cfg.attention_dropout

        # Multi-head self-attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.village_feature_dim,
            num_heads=self.num_attention_heads,
            dropout=self.attention_dropout,
            batch_first=True
        )

        # Simple feedforward sub-block
        self.feedforward = nn.Sequential(
            nn.Linear(self.village_feature_dim, self.feedforward_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.feedforward_hidden_dim, self.village_feature_dim),
        )

        self.layernorm1 = nn.LayerNorm(self.village_feature_dim)
        self.layernorm2 = nn.LayerNorm(self.village_feature_dim)
        self.dropout_layer = nn.Dropout(self.attention_dropout)

    def forward(self, x):
        """
        x: shape [batch_size, num_villages, village_feature_dim]
        """
        # Self-attention
        attn_out, _ = self.multi_head_attention(x, x, x)
        x = x + self.dropout_layer(attn_out)
        x = self.layernorm1(x)

        # Feedforward
        ff_out = self.feedforward(x)
        ff_out = self.dropout_layer(ff_out)

        x = x + ff_out
        x = self.layernorm2(x)
        return x  # [batch_size, num_villages, village_feature_dim]


###############################################################################
# VillagePolicy
#  - Stage 1: global attention => selection logits
#  - pick K villages => flatten => Stage 2 => MLP => produce K * num_modes
###############################################################################
class VillagePolicy(nn.Module):
    """
    Two-stage policy for selecting 'village_per_step' villages (Stage 1)
    and then picking a renovation mode for each (Stage 2).

    Stage 1 (forward_selection):
      - A multi-head self-attention block over all villages => selection logits [B, N].
    Stage 2 (forward_modes):
      - Takes the K selected villages' embeddings => flatten => produce [B, K * num_modes],
        reshape => [B, K, num_modes].
    """

    def __init__(self, cfg):
        """
        Expecting cfg to define:
          - cfg.village_feature_dim (int): dimension of each village embedding
          - cfg.num_attention_heads (int)
          - cfg.feedforward_hidden_dim (int)
          - cfg.attention_dropout (float)
          - cfg.selection_hidden_dim (int) used for a final MLP or you can skip
          - cfg.mode_hidden_dim (int)
          - cfg.num_combinations = len(cfg.combinations)
          - cfg.num_far_values  = len(cfg.FAR_values)
          - cfg.village_per_step
          - possibly more
        """
        super().__init__()
        self.village_feature_dim = cfg.village_feature_dim

        # Basic problem dimension
        self.num_combinations = len(cfg.combinations)
        self.num_far_values   = len(cfg.FAR_values)
        self.num_modes        = self.num_combinations * self.num_far_values
        self.village_per_step = cfg.village_per_step

        # --- Stage 1: Selection via attention ---
        self.attention_block_selection = SimpleSelfAttentionBlock(cfg)
        # Final MLP for selection logits
        # For simplicity, we define a small hidden layer or direct => 1
        self.selection_mlp = self.create_policy_head(cfg.village_feature_dim, cfg.hidden_dims, name="selection")

        # --- Stage 2: Mode selection MLP ---
        # We'll flatten K * village_feature_dim => pass MLP => K * num_modes
        self.mode_hidden_dim = cfg.mode_hidden_dim
        # a small feedforward
        self.mode_mlp = nn.Sequential(
            nn.Linear(self.village_per_step * self.village_feature_dim, self.mode_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mode_hidden_dim, self.village_per_step * self.num_modes)
        )

    def create_policy_head(self, input_size, hidden_size, name):
        """Create the policy head."""
        policy_head = nn.Sequential()
        for i in range(len(hidden_size)):
            if i == 0:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(input_size, hidden_size[i])
                )
            else:
                policy_head.add_module(
                    '{}_linear_{}'.format(name, i),
                    nn.Linear(hidden_size[i - 1], hidden_size[i], bias=False)
                )
            policy_head.add_module(
                '{}_relu_{}'.format(name, i),
                nn.LeakyReLU()
            )
        return policy_head

    def scale_logits(self, logits: torch.Tensor, limit: float=15.0, dim=-1):
        """
        Scales each row so that max abs value does not exceed 'limit'.
        If max_abs < limit, no change.
        """
        max_abs = torch.max(torch.abs(logits), dim=dim, keepdim=True).values
        scale_factor = torch.clamp(max_abs / limit, min=1.0)
        return logits / scale_factor

    ########################################################################
    # Stage 1: forward_selection
    ########################################################################
    def forward_selection(self, village_embeddings: torch.Tensor):
        """
        village_embeddings: [B, N, village_feature_dim]
        Return:
          selection_logits: [B, N]
          attention_out: [B, N, village_feature_dim]  (contextual embeddings)
        """
        # pass through attention
        attn_out = self.attention_block_selection(village_embeddings)
        B, N, D = attn_out.shape

        # flatten => pass selection_mlp => [B*N, 1] => reshape
        flat_attn = attn_out.view(B*N, D)
        selection_scores = self.selection_mlp(flat_attn).view(B, N)

        selection_scores = self.scale_logits(selection_scores, dim=-1)
        return selection_scores, attn_out

    ########################################################################
    # Stage 2: forward_modes
    ########################################################################
    def forward_modes(self, chosen_subset: torch.Tensor):
        """
        chosen_subset: shape [B, K, village_feature_dim] 
          (the K chosen villages' embeddings from attention_out).
        We flatten => shape [B, K*village_feature_dim] => pass an MLP => shape [B, K*num_modes].
        Then reshape => [B, K, num_modes].
        """
        B, K, D = chosen_subset.shape  # K = self.village_per_step
        # flatten
        flattened = chosen_subset.view(B, K*D)
        # pass MLP
        mode_scores = self.mode_mlp(flattened)  # [B, K*num_modes]
        # reshape
        mode_scores = mode_scores.view(B, K, self.num_modes)
        mode_scores = self.scale_logits(mode_scores, dim=-1)
        return mode_scores

    ########################################################################
    # select_action:
    ########################################################################
    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action=False):
        """
        Return:
          actions_list [B][village_per_step] => (v_idx, comb_idx, far_idx)
          log_prob => [B]
          entropy  => [B]
        """
        B, N, D = village_embeddings.shape
        selection_logits, attn_out = self.forward_selection(village_embeddings)

        batch_actions = []
        all_log_probs = []
        all_entropies = []

        for b in range(B):
            logits_b = selection_logits[b]  # [N]
            if mean_action:
                # pick top-K
                _, top_idx = torch.topk(logits_b, k=self.village_per_step, dim=0)
                chosen_vs = top_idx
                selection_lp_b = torch.zeros(1, device=logits_b.device)
                selection_ent_b= torch.zeros(1, device=logits_b.device)
            else:
                # sample K from softmax
                probs_b = F.softmax(logits_b, dim=-1)
                chosen_vs = torch.multinomial(probs_b, self.village_per_step, replacement=False)
                log_probs_b = torch.log(probs_b[chosen_vs])
                selection_lp_b = log_probs_b.sum()
                selection_ent_b= torch.zeros(1, device=logits_b.device)

            # gather embeddings => shape [K, village_feature_dim], unsqueeze batch
            subset_emb = attn_out[b, chosen_vs, :].unsqueeze(0) # [1,K,dim]
            # pass forward_modes => [1,K,num_modes]
            mode_scores = self.forward_modes(subset_emb)
            if mean_action:
                chosen_modes = torch.argmax(mode_scores, dim=-1)  # shape [1,K]
                dist_b = Categorical(logits=mode_scores)
                mode_lps = dist_b.log_prob(chosen_modes)  # [1,K]
                mode_ents= dist_b.entropy()               # [1,K]
            else:
                dist_b = Categorical(logits=mode_scores)
                chosen_modes = dist_b.sample()            # [1,K]
                mode_lps = dist_b.log_prob(chosen_modes)  # [1,K]
                mode_ents= dist_b.entropy()               # [1,K]

            total_lp_b = selection_lp_b + mode_lps.sum()
            total_ent_b= selection_ent_b + mode_ents.sum()

            # build final action
            actions_b = []
            chosen_modes_b = chosen_modes[0]  # shape [K]
            for i in range(self.village_per_step):
                v_idx = chosen_vs[i].item()
                m_idx = chosen_modes_b[i].item()
                comb_idx = m_idx // self.num_far_values
                far_idx  = m_idx % self.num_far_values
                actions_b.append((v_idx, comb_idx, far_idx))

            batch_actions.append(actions_b)
            all_log_probs.append(total_lp_b)
            all_entropies.append(total_ent_b)

        log_prob = torch.stack(all_log_probs, dim=0)
        entropy = torch.stack(all_entropies, dim=0)
        return batch_actions, log_prob, entropy

    ########################################################################
    # get_log_prob_entropy:
    ########################################################################
    def get_log_prob_entropy(self, village_embeddings: torch.Tensor, batch_actions, mean_action=False):
        """
        batch_actions: [B][village_per_step] => (v_idx, comb_idx, far_idx)
        Return total_log_prob [B], total_entropy [B].
        """
        B, N, D = village_embeddings.shape
        selection_logits, attn_out = self.forward_selection(village_embeddings)

        all_lp = []
        all_ent= []

        for b in range(B):
            acts_b = batch_actions[b]
            if mean_action:
                sel_lp_b = torch.zeros(1, device=selection_logits.device)
                sel_ent_b= torch.zeros(1, device=selection_logits.device)
            else:
                # approximate sum of log p for the chosen villages
                probs_b = F.softmax(selection_logits[b], dim=-1)  # [N]
                chosen_lps = []
                for (v_idx, c_idx, f_idx) in acts_b:
                    chosen_lps.append(torch.log(probs_b[v_idx]))
                sel_lp_b = torch.stack(chosen_lps).sum()
                sel_ent_b= torch.zeros(1, device=selection_logits.device)

            chosen_vs = [a[0] for a in acts_b]
            chosen_vs_t = torch.tensor(chosen_vs, device=attn_out.device, dtype=torch.long)
            subset_emb = attn_out[b, chosen_vs_t, :].unsqueeze(0) # [1, K, dim]

            mode_scores = self.forward_modes(subset_emb)  # [1, K, num_modes]

            mode_lp_list = []
            mode_ent_list= []
            for (v_idx, c_idx, f_idx) in acts_b:
                # find index in chosen_vs_t:
                sub_idx = (v_idx==chosen_vs_t).nonzero(as_tuple=True)[0]
                i_sub   = sub_idx.item()
                dist_v  = Categorical(logits=mode_scores[0, i_sub])
                mode_id = c_idx * self.num_far_values + f_idx

                lp_v = dist_v.log_prob(torch.tensor(mode_id, device=mode_scores.device))
                ent_v= dist_v.entropy()
                mode_lp_list.append(lp_v)
                mode_ent_list.append(ent_v)

            mode_lp_b = torch.stack(mode_lp_list).sum()
            mode_ent_b= torch.stack(mode_ent_list).sum()

            total_lp_b = sel_lp_b + mode_lp_b
            total_ent_b= sel_ent_b + mode_ent_b

            all_lp.append(total_lp_b)
            all_ent.append(total_ent_b)

        total_lp = torch.stack(all_lp, dim=0)
        total_ent= torch.stack(all_ent, dim=0)
        return total_lp, total_ent

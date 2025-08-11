import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.config import Config

class VillagePolicy(nn.Module):
    """
    A policy network for PPO that:
      - Takes per-village embeddings [B, N, embed_dim].
      - Produces:
          1) selection_logits: [B, N] (which villages to pick)
          2) mode_logits:      [B, N, num_modes] (which renovation mode to assign)
      - Allows you to select 'village_per_step' villages either by:
         a) mean_action=True  => pick top-k + argmax mode
         b) mean_action=False => sample k distinct villages w/ a softmax distribution & sample modes
      - For the sampling case, we do a simplified log-prob for selection by summing log p_i 
        of the chosen villages (an approximation for no-replacement sampling).
      - Instead of returning a dict, returns a list-of-lists-of-tuples: [B][k](village_idx, comb, far).
    """

    def __init__(self,cfg: Config):
        super().__init__()
        self.num_comb = len(cfg.combinations)
        self.num_far = len(cfg.FAR_values)
        self.num_modes = self.num_comb * self.num_far
        self.village_per_step = cfg.village_per_step
        self.hidden_dims = cfg.policy_hidden_dim

        # A small MLP to process each village embedding.
        
        self.shared_mlp = self.create_policy_head(cfg.village_feature_dim, self.hidden_dims, "policy")
        # Outputs:
        #   1) A scalar for selection
        self.select_head = nn.Linear(self.hidden_dims[-1], 1)
        #   2) A vector of size num_modes for renovation modes
        self.mode_head = nn.Linear(self.hidden_dims[-1], self.num_modes)

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

    def scale_logits(self, logits: torch.Tensor, limit: float = 10.0, dim = -1) -> torch.Tensor:
        """
        Scales each 'row' of logits along dimension -1 so that the maximum absolute value
        does not exceed `limit`. If the max absolute value is already below `limit`, no change.

        Args:
            logits_to_scale (torch.Tensor): Arbitrary shape [..., D], where D is the last dimension.
            limit (float): The desired maximum absolute value L.

        Returns:
            torch.Tensor: A new tensor with the same shape, but scaled in the last dimension
                        so that values lie in [-limit, limit].
        """
        # 1) Compute max of the absolute values along the last dimension.
        #    shape of max_abs -> [..., 1] (keeping dimension for easy broadcast).
        max_abs = torch.max(torch.abs(logits), dim=dim, keepdim=True).values

        # 2) Compute the scale factor as (max_abs / limit).
        #    If max_abs <= limit, scale_factor <= 1 => no scaling needed.
        #    So we clamp scale_factor to be >= 1.0 so that it doesn't reduce the scale if below limit.
        scale_factor = torch.clamp(max_abs / limit, min=1.0)

        # 3) Finally, divide the logits by this scale_factor (broadcast along last dimension).
        scaled_logits = logits / scale_factor

        return scaled_logits

    def forward(self, village_embeddings: torch.Tensor):
        """
        Forward pass.

        Args:
            village_embeddings: [B, N, embed_dim]
                - B = batch size
                - N = total villages (e.g. 1400)
                - embed_dim = dimension of each village embedding (e.g. 128)

        Returns:
            selection_logits: [B, N]
            mode_logits:      [B, N, num_modes]
        """
        B, N, E = village_embeddings.shape

        # Flatten (B,N,E) -> (B*N, E)
        x = village_embeddings.reshape(B * N, E)
        h = self.shared_mlp(x)  # [B*N, hidden_dim]

        select_logit = self.select_head(h)    # [B*N, 1]
        mode_logit   = self.mode_head(h)      # [B*N, self.num_modes]

        # Reshape back
        selection_logits = select_logit.view(B, N)               # [B, N]
        mode_logits      = mode_logit.view(B, N, self.num_modes) # [B, N, num_modes]

        selection_logits = self.scale_logits(selection_logits, dim = -1)
        mode_logits      = self.scale_logits(mode_logits, dim = -1)

        return selection_logits, mode_logits

    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action = False):
        """
        Given per-village embeddings, select 'village_per_step' villages and pick a mode.

        If mean_action=True:
          - Greedily pick top-k by selection_logits
          - For each selected village, choose argmax from mode_logits

        If mean_action=False (sampling):
          - Convert selection_logits to probabilities (softmax)
          - Sample k distinct villages with torch.multinomial(..., replacement=False)
          - Approximate selection log-prob = sum(log p_i) for chosen i
          - For each selected village, sample from the mode distribution (softmax of mode_logits)

        Returns:
            actions_list: a list of length B,
              where actions_list[b] is a list of length village_per_step with tuples:
                (village_idx, combination_idx, far_idx)
            log_prob: [B]   - total log-prob of chosen subset + modes
            entropy:  [B]   - total entropy from the distributions
        """
        # print(mean_action)
        B, N, E = village_embeddings.shape
        selection_logits, mode_logits = self.forward(village_embeddings)

        # We'll accumulate results here
        batched_actions = []  # list-of-lists-of-tuples
        all_log_probs = []
        all_entropies = []

        for b in range(B):
            sel_logits_b = selection_logits[b]   # [N]
            mode_logits_b = mode_logits[b]       # [N, num_modes]
            # 1) Village Selection
            if mean_action:
                # Greedily pick top-k
                _, top_indices_b = torch.topk(sel_logits_b, k=self.village_per_step, dim=0)
                selected_villages_b = top_indices_b
                # In "greedy" case, approximate selection log-prob = 0
                selection_lp_b = torch.zeros(1, device=sel_logits_b.device)
                selection_ent_b = torch.zeros(1, device=sel_logits_b.device)

            else:
                # Sample k distinct villages from softmax distribution
                sel_probs_b = F.softmax(sel_logits_b, dim=-1)  # [N]
                selected_villages_b = torch.multinomial(
                    sel_probs_b,
                    self.village_per_step,
                    replacement=False
                )  # shape [village_per_step]

                # Approx. log-prob of that set by summing log p_i for chosen i
                log_probs_b = torch.log(sel_probs_b[selected_villages_b])  # [village_per_step]
                selection_lp_b = log_probs_b.sum()
                selection_ent_b = torch.zeros(1, device=sel_logits_b.device)

            # 2) Mode Choice for each selected village
            chosen_mode_logits_b = mode_logits_b[selected_villages_b, :]  # [village_per_step, num_modes]

            if mean_action:
                # Argmax per selected village
                chosen_modes_b = torch.argmax(chosen_mode_logits_b, dim=-1)  # [village_per_step]
                # For log-prob & entropy, get them from a Categorical
                dist_b = Categorical(logits=chosen_mode_logits_b)
                mode_log_probs_b = dist_b.log_prob(chosen_modes_b)  # [village_per_step]
                mode_entropies_b = dist_b.entropy()                 # [village_per_step]
            else:
                # Sample from the distribution for each selected village
                dist_b = Categorical(logits=chosen_mode_logits_b)
                chosen_modes_b = dist_b.sample()                    # [village_per_step]
                mode_log_probs_b = dist_b.log_prob(chosen_modes_b)  # [village_per_step]
                mode_entropies_b = dist_b.entropy()                 # [village_per_step]

            # Sum up logs & entropies
            total_lp_b = selection_lp_b + mode_log_probs_b.sum()
            total_ent_b = selection_ent_b + mode_entropies_b.sum()

            # Build the action list of tuples (village_idx, combination_idx, far_idx)
            actions_b = []
            for i in range(self.village_per_step):
                v_idx = selected_villages_b[i].item()
                mode_idx = chosen_modes_b[i].item()
                comb_idx = mode_idx // self.num_far
                far_idx = mode_idx % self.num_far
                actions_b.append((v_idx, comb_idx, far_idx))

            batched_actions.append(actions_b)
            all_log_probs.append(total_lp_b)
            all_entropies.append(total_ent_b)

        log_prob = torch.stack(all_log_probs, dim=0)  # [B]
        entropy = torch.stack(all_entropies, dim=0)   # [B]

        return batched_actions, log_prob, entropy

    def get_log_prob_entropy(self, village_embeddings: torch.Tensor, batched_actions, mean_action=False):
        """
        Compute log-prob and entropy of the given 'batched_actions' under the current policy.

        batched_actions is [B][village_per_step] of tuples (village_idx, comb_idx, far_idx).

        If mean_action=True, we can't easily define a distribution for top-k => selection log-prob=0.
        If mean_action=False, we approximate selection log-prob as sum(log p_i) for chosen i,
          ignoring no-replacement factor.

        Args:
            village_embeddings: [B, N, embed_dim]
            batched_actions: list-of-lists-of-tuples
                e.g. batched_actions[b][i] = (village_idx, comb_idx, far_idx)
            mean_action: bool

        Returns:
            total_log_prob: [B]
            total_entropy:  [B]
        """
        B, N, E = village_embeddings.shape
        selection_logits, mode_logits = self.forward(village_embeddings)
        # print(mode_logits)

        all_log_probs = []
        all_entropies = []

        for b in range(B):
            actions_b = batched_actions[b]  # list of (v_idx, comb_idx, far_idx) length = village_per_step

            if mean_action:
                # 0 for selection
                selection_lp_b = torch.zeros(1, device=village_embeddings.device)
                selection_ent_b = torch.zeros(1, device=village_embeddings.device)
            else:
                # sum-of-log-probs for the chosen villages
                sel_logits_b = selection_logits[b]  # [N]
                sel_probs_b = F.softmax(sel_logits_b, dim=-1)  # [N]

                # gather log probs for each chosen village
                chosen_log_probs = []
                for (v_idx, comb_idx, far_idx) in actions_b:
                    chosen_log_probs.append(torch.log(sel_probs_b[v_idx]))

                selection_lp_b = torch.stack(chosen_log_probs).sum()
                selection_ent_b = torch.zeros(1, device=village_embeddings.device)

            # For modes
            mode_lp_b = []
            mode_ent_b = []
            for (v_idx, comb_idx, far_idx) in actions_b:
                # Recompute the distribution for that village
                logits_v = mode_logits[b, v_idx, :]  # [num_modes]
                dist_v = Categorical(logits=logits_v)

                # mode index
                mode_idx = comb_idx * self.num_far + far_idx
                lp_v = dist_v.log_prob(torch.tensor(mode_idx, device=logits_v.device))
                ent_v = dist_v.entropy()  # scalar
                mode_lp_b.append(lp_v)
                mode_ent_b.append(ent_v)

            mode_lp_b = torch.stack(mode_lp_b, dim=0).sum()     # sum over selected villages
            mode_ent_b = torch.stack(mode_ent_b, dim=0).sum()

            total_lp_b = selection_lp_b + mode_lp_b
            total_ent_b = selection_ent_b + mode_ent_b

            all_log_probs.append(total_lp_b)
            all_entropies.append(total_ent_b)

        total_log_prob = torch.stack(all_log_probs, dim=0)
        total_entropy = torch.stack(all_entropies, dim=0)

        return total_log_prob, total_entropy
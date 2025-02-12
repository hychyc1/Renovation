from torch import nn
import torch
import torch.nn.functional as F

class PolicyHead(nn.Module):
    def __init__(self, cfg):
        """
        Policy Head for Grid, Combination, and Strength Selection, with PPO utilities.

        Args:
        - cfg (object): Configuration object containing the following attributes:
            - cfg.feature_dim (int): Dimension of the input state encoding.
            - cfg.grid_size (tuple): Dimensions of the grid (n, m).
            - cfg.num_comb (int): Number of possible combinations.
            - cfg.num_far (int): Number of possible strength levels.
        """
        super().__init__()
        self.feature_dim = cfg.feature_dim
        self.grid_size = cfg.grid_size
        self.num_comb = len(cfg.combinations)
        self.num_far = len(cfg.FAR_values)
        self.grid_per_year = cfg.grid_per_year

        self.n, self.m = self.grid_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_dim, 256)
        self.fc2 = nn.Linear(256, self.n * self.m * (1 + self.num_comb + self.num_far))

        # Softmax for grid, combination, and strength probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state_encoding):
        """
        Forward pass through the policy head.

        Args:
        - state_encoding (torch.Tensor): Encoded state, shape [batch_size, feature_dim].

        Returns:
        - policy_output (torch.Tensor): Output of the policy, 
                                        shape [batch_size, n, m, 1 + num_comb + num_far].
        """
        # Fully connected layers
        x = F.relu(self.fc1(state_encoding))
        x = self.fc2(x)

        # Reshape output
        x = x.view(-1, self.n, self.m, 1 + self.num_comb + self.num_far)

        # Separate channels
        grid_probs = self.softmax(x[..., 0].flatten(1)).view(-1, self.n, self.m, 1)  # Grid probabilities
        comb_probs = self.softmax(x[..., 1:1 + self.num_comb])  # Combination probabilities
        far_probs = self.softmax(x[..., 1 + self.num_comb:])  # Strength probabilities

        return torch.cat([grid_probs, comb_probs, far_probs], dim=-1)

    def select_action(self, x, mask, mean_action=False):
        """
        Selects actions for grid, combination, and strength selection.

        Args:
        - x (torch.Tensor): Output from the state encoder, shape [batch_size, feature_dim].
        - mask (torch.Tensor): Valid grid mask (AREA > 0), shape [n, m].
        - mean_action (bool): If True, select top `grid_per_year` grids with highest probabilities.
                            If False, sample `grid_per_year` grids based on probabilities.

        Returns:
        - actions (list of list of tuples): For each batch, a list of `grid_per_year` tuples (i, j, combination, strength).
        """
        # Forward pass through the policy head
        policy_output = self.forward(x)  # Shape: [batch_size, n, m, 1 + num_comb + num_far]

        # Extract components
        grid_probs = policy_output[..., 0]  # Shape: [batch_size, n, m]
        comb_probs = policy_output[..., 1:1 + self.num_comb]  # Shape: [batch_size, n, m, num_comb]
        far_probs = policy_output[..., 1 + self.num_comb:]  # Shape: [batch_size, n, m, num_far]

        batch_size, n, m = grid_probs.shape
        actions = []

        for b in range(batch_size):
            # Apply mask to grid probabilities
            invalid_value = -2 ** 32  # Large negative value for invalid grids
            masked_grid_probs = torch.where(mask, grid_probs[b], torch.full_like(grid_probs[b], invalid_value))

            # Flatten masked grid probabilities
            prob_flat = masked_grid_probs.flatten()  # Flatten grid probabilities: [n * m]
            comb_probs_flat = comb_probs[b].view(n * m, self.num_comb)  # Flatten combination probs: [n * m, num_comb]
            far_probs_flat = far_probs[b].view(n * m, self.num_far)  # Flatten strength probs: [n * m, num_far]

            # print(F.softmax(prob_flat, dim=0))
            if mean_action:
                # Select the top `grid_per_year` grid probabilities
                top_indices = torch.topk(prob_flat, k=self.grid_per_year).indices
            else:
                # Sample `grid_per_year` grids based on probabilities
                top_indices = torch.multinomial(F.softmax(prob_flat, dim=0), num_samples=self.grid_per_year, replacement=False)

            batch_actions = []
            for idx in top_indices:
                i, j = divmod(idx.item(), m)  # Convert flat index to grid indices
                if mean_action:
                    combination = torch.topk(comb_probs_flat[idx], k=1).indices
                    strength = torch.topk(far_probs_flat[idx], k=1).indices 
                else:
                    combination = torch.multinomial(comb_probs_flat[idx], num_samples=1).item()  # Sample combination
                    strength = torch.multinomial(far_probs_flat[idx], num_samples=1).item()  # Sample strength
                batch_actions.append((i, j, combination, strength))

            actions.append(batch_actions)

        return actions


    def get_log_prob_entropy(self, x, actions, masks):
        """
        Computes the log probabilities and entropy for a batch of grid, combination, and strength actions.

        Args:
        - x (torch.Tensor): Output from the state encoder, shape [batch_size, feature_dim].
        - actions (list of list of tuples): Actions selected for each batch. 
                                            Each action is (i, j, combination, strength).
        - mask (torch.Tensor): Valid grid mask (AREA > 0), shape [batch_size, n, m].

        Returns:
        - log_probs (torch.Tensor): Log probabilities of the actions, shape [batch_size, num_actions].
        - entropy (torch.Tensor): Entropy of the policy, shape [batch_size].
        """
        # Forward pass through the policy head
        policy_output = self.forward(x)  # Shape: [batch_size, n, m, 1 + num_comb + num_far]

        # Separate probabilities
        grid_probs = policy_output[..., 0]  # Shape: [batch_size, n, m]
        comb_probs = policy_output[..., 1:1 + self.num_comb]  # Shape: [batch_size, n, m, num_comb]
        far_probs = policy_output[..., 1 + self.num_comb:]  # Shape: [batch_size, n, m, num_far]

        batch_size, n, m = grid_probs.shape
        num_actions = len(actions[0])  # Number of actions taken per batch (e.g., 10)

        log_probs = []
        entropies = []

        grid_probs_padding = torch.full_like(grid_probs[0], -2 ** 32) # Large negative value for invalid grids

        for b in range(batch_size):
            # Apply mask to grid probabilities
            invalid_value = -2 ** 32  
            # masked_grid_probs = torch.where(masks[b], grid_probs[b], grid_probs_padding)
            # using masks here results in many nan. Figure out why.
            masked_grid_probs = grid_probs[b]

            prob_flat = masked_grid_probs.flatten()  # Flatten grid probabilities: [n * m]
            comb_probs_flat = comb_probs[b].view(n * m, self.num_comb)  # Flatten combination probs: [n * m, num_comb]
            far_probs_flat = far_probs[b].view(n * m, self.num_far)  # Flatten strength probs: [n * m, num_far]

            batch_log_probs = []
            batch_entropy = 0.0
            for action in actions[b]:
                i, j, combination, strength = action  # Extract grid, combination, and strength

                # Log probability for the grid
                flat_idx = i * m + j  # Convert (i, j) to flat index
                log_prob_grid = torch.log(prob_flat[flat_idx] + 1e-10)  # Avoid log(0) with epsilon

                # Log probability for the combination
                log_prob_comb = torch.log(comb_probs_flat[flat_idx, combination] + 1e-10)

                # Log probability for the strength
                log_prob_far = torch.log(far_probs_flat[flat_idx, strength] + 1e-10)

                # Combine log probabilities
                batch_log_probs.append(log_prob_grid + log_prob_comb + log_prob_far)

                # Entropy
                batch_entropy += -prob_flat[flat_idx] * log_prob_grid  # Grid entropy
                batch_entropy += -(comb_probs_flat[flat_idx] * torch.log(comb_probs_flat[flat_idx] + 1e-10)).sum()  # Combination entropy
                batch_entropy += -(far_probs_flat[flat_idx] * torch.log(far_probs_flat[flat_idx] + 1e-10)).sum()  # Strength entropy

            log_probs.append(torch.stack(batch_log_probs))
            entropies.append(batch_entropy)

        # Stack results for the batch
        log_probs = torch.stack(log_probs)  # Shape: [batch_size, num_actions]
        entropy = torch.stack(entropies)  # Shape: [batch_size]

        return log_probs, entropy


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Config.config import Config

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

        # A small MLP to process each village embedding.
        self.shared_mlp = nn.Sequential(
            nn.Linear(cfg.village_feature_dim, cfg.policy_hidden_dim),
            nn.ReLU(),
        )
        # Outputs:
        #   1) A scalar for selection
        self.select_head = nn.Linear(self.hidden_dim, 1)
        #   2) A vector of size num_modes for renovation modes
        self.mode_head = nn.Linear(self.hidden_dim, self.num_modes)

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

        return selection_logits, mode_logits

    @torch.no_grad()
    def select_action(self, village_embeddings: torch.Tensor, mean_action: bool = False):
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
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

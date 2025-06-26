import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from models.policy import VillagePolicy
from models.state_encoder_cnn import FeaturePyramidEncoder
from models.state_encoder_gnn import FeatureEncoderGNN
from models.value import ValueNetwork
from utils.config import Config
from tqdm import tqdm


class PPOAgent:
    def __init__(self, env, cfg: Config, device, log_file="training.log"):
        """
        PPO Agent for training with the provided environment.

        Args:
            env: The environment with `reset` and `step` methods.
            cfg: Configuration object (hyperparams, etc.).
            device: A torch.device (e.g. torch.device("cuda") or "cpu")
            log_file: Path to the log file for training logs.
        """
        self.env = env
        self.cfg = cfg
        self.device = device  # e.g. torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = cfg.name

        # Initialize networks
        self.policy_net = VillagePolicy(cfg).to(self.device)
        self.value_net = ValueNetwork(cfg).to(self.device)
        if cfg.state_encoder_type == 'CNN':
            self.state_encoder = FeaturePyramidEncoder(cfg).to(self.device)
        elif cfg.state_encoder_type == 'GNN':
            self.state_encoder = FeatureEncoderGNN(cfg).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)

        # PPO hyperparameters
        self.gamma = cfg.gamma
        self.eps = cfg.eps
        self.entropy_coef = cfg.entropy_coef
        self.value_pred_coef = cfg.value_pred_coef
        self.batch_size = cfg.batch_size
        self.num_epochs = cfg.num_epochs
        self.num_episodes_per_iteration = cfg.num_episodes_per_iteration

        # Set up logging
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

    def parse_state(self, state):
        """
        Converts the states into 2 tensors on self.device.
        Args:
            state = (grid, villages, step)
            grid (dict): k keys, each is a 2D array (n, m).
            villages (ndarray): shape [num_villages, 4]

        Returns:
            grid_tensor: shape (k, n, m) on device
            village_tensor: shape (num_villages, 3) on device, drop the last one since it is ID
            year: shape (1) on device
        """
        grid, villages, year = state
        # print(state)
        # print(villages)
        grid_tensor = torch.stack(list(grid.values()), axis=0)
        village_tensor = torch.tensor(villages, dtype=torch.float32, device=self.device)
        # print(village_tensor.shape)
        village_tensor = village_tensor[:,:3]
        # print(village_tensor.shape)
        year_tensor = torch.tensor(year, device=self.device)
        return grid_tensor, village_tensor, year_tensor

    def collect_trajectory(self, mean_action=False, info_list=None):
        """
        Collects a single-episode trajectory by interacting with the environment.

        Args:
            mean_action (bool): If True, selects deterministic (argmax) actions.
            info_list (list): Optional list to store env info dicts.

        Returns:
            states (list): Each element is (village_feats, global_feats) on device
            actions (list): Each action (v_idx, comb_idx, far_idx)
            rewards (list): Rewards at each step
            log_probs (list): Log probabilities of each action
            dones (list): Done flags
        """
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        state = self.env.reset()
        done = False

        while not done:
            # Convert state to device
            grid_tensor, village_tensor, year_tensor = self.parse_state(state)

            # Encode
            with torch.no_grad():
                v_feats, g_feats = self.state_encoder(
                    grid_tensor.unsqueeze(0),  # add batch dim
                    village_tensor.unsqueeze(0),
                    year_tensor.unsqueeze(0)
                )
                # v_feats: [1, N, embed_dim], g_feats: [1, global_dim]

                # Policy picks an action
                selected_actions, log_prob, _ = self.policy_net.select_action(
                    v_feats, mean_action
                )
                # selected_actions, log_prob: shape [1], each a list-of-tuples or similar

            # Step environment (move action to CPU/np if needed)
            action_for_env = selected_actions[0]  # presumably a Python tuple
            next_state, reward, done, info = self.env.step(action_for_env)

            if info_list is not None:
                info_list.append(info)

            # Save transition
            states.append((v_feats.squeeze(0), g_feats.squeeze(0)))  # shape [N, embed_dim], [global_dim]
            actions.append(action_for_env)
            rewards.append(reward)
            log_probs.append(log_prob[0])  # log_prob is shape [1], so log_prob[0]
            dones.append(done)

            state = next_state

        return states, actions, rewards, log_probs, dones

    def compute_advantages(self, rewards, values, dones):
        """
        Computes advantages using a simple (one-step style) GAE approach.
        (Check if you want a full multi-step GAE with next-state values.)

        Args:
            rewards (list of float)
            values (list of float) or np array
            dones (list of bool)

        Returns:
            advantages (torch.Tensor) [T]
            returns (torch.Tensor) [T]
        """
        advantages = []
        returns = []
        next_value = 0
        advantage = 0

        # We move everything in reverse
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                next_value = 0
            delta = reward + self.gamma * next_value - value
            advantage = delta + self.gamma * (1 - done) * advantage

            returns.insert(0, reward + self.gamma * next_value)
            advantages.insert(0, advantage)
            next_value = value

        # Return them as Tensors on device
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return advantages, returns

    def shuffle_trajectory(self, states, actions, log_probs, advantages, returns):
        """
        Shuffle transitions to reduce correlation.

        We assume each is a list of length T except advantages/returns are T-dim Tensors.
        We'll keep advantages/returns as Tensors, so we must rearrange them with the same indices.
        """
        n = len(states)
        idxs = torch.randperm(n, device=self.device)  # create a random perm on device
        # Convert idxs to CPU if we want to do python indexing
        idxs_cpu = idxs.cpu().numpy()

        states_shuffled = [states[i] for i in idxs_cpu]
        actions_shuffled = [actions[i] for i in idxs_cpu]
        log_probs_shuffled = [log_probs[i] for i in idxs_cpu]
        advantages_shuffled = advantages[idxs]
        returns_shuffled = returns[idxs]
        return states_shuffled, actions_shuffled, log_probs_shuffled, advantages_shuffled, returns_shuffled

    def update_policy(self, village_features, actions, log_probs_old, advantages):
        """
        Updates the policy net using PPO's clipped surrogate objective.
        village_features: [T, ...] on self.device
        actions: list of length T
        log_probs_old: list of length T (or tensor)
        advantages: shape [T]
        """
        # Convert log_probs_old if it's a list to a tensor on device
        log_probs_old = torch.stack(log_probs_old).to(self.device)

        T = len(village_features)
        for _ in range(self.num_epochs):
            # minibatch
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                vf_batch = village_features[start:end]  # shape [MB, ...]
                act_batch = actions[start:end]
                lpo_batch = log_probs_old[start:end]  # [MB]
                adv_batch = advantages[start:end]      # [MB]

                # Recompute log_probs_new
                log_probs_new, entropy = self.policy_net.get_log_prob_entropy(
                    vf_batch, act_batch
                )
                # ratio
                ratio = torch.exp(log_probs_new - lpo_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

    def update_value(self, global_features, returns):
        """
        MSE loss for value function.
        global_features: [T, g_dim] on device
        returns: [T] on device
        """
        T = len(global_features)
        for _ in range(self.num_epochs):
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                gf_batch = global_features[start:end]
                ret_batch = returns[start:end]

                values = self.value_net(gf_batch).squeeze(-1)
                value_loss = self.value_pred_coef * (values - ret_batch).pow(2).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def save_checkpoint(self, file_path):
        file_path = "checkpoints/" + file_path
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'state_encoder': self.state_encoder.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, file_path)
        print(f"Checkpoint saved to {file_path}", flush=True)
        self.logger.info(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.logger.info(f"Checkpoint loaded from {file_path}")

    def train(self, num_iterations):
        print(f"Prepared to train {num_iterations} iterations", flush=True)
        for iteration in range(num_iterations):
            print(f"Training Iteration {iteration}", flush=True)
            states, actions, rewards, log_probs, dones = [], [], [], [], []
            info = []
            for episode in range(self.num_episodes_per_iteration):
                if episode % 100 == 0:
                    print(f"Collecting Episode {episode}", flush=True)
                cur_states, cur_actions, cur_rewards, cur_log_probs, cur_dones = self.collect_trajectory(info_list=info)
                states.extend(cur_states)
                actions.extend(cur_actions)
                rewards.extend(cur_rewards)
                log_probs.extend(cur_log_probs)
                dones.extend(cur_dones)
                # print(cur_rewards)
            
            print(f"Finish Collecting Episodes", flush=True)

            # States is list of (village_feats, global_feats)
            # Convert to Tensors on device
            _, global_feats_list = zip(*states)  # unzips into two tuples
            global_feats_tensor = torch.stack(global_feats_list).to(self.device)     # [T, global_dim]

            # Compute values => for advantage
            with torch.no_grad():
                values_t = self.value_net(global_feats_tensor).squeeze(-1)  # shape [T]
            values_np = values_t.cpu().numpy()  # if we want to do python loop in compute_advantages

            # Compute advantages
            advantages, returns = self.compute_advantages(rewards, values_np, dones)

            # Shuffle trajectory
            states_shuf, actions_shuf, log_probs_shuf, adv_shuf, ret_shuf = self.shuffle_trajectory(
                states, actions, log_probs, advantages, returns
            )

            # Rebuild Tensors from shuffled states
            vf_shuf_list, gf_shuf_list = zip(*states_shuf)
            vf_shuf = torch.stack(vf_shuf_list).to(self.device)
            gf_shuf = torch.stack(gf_shuf_list).to(self.device)
            log_probs_shuf_t = [lp if torch.is_tensor(lp) else torch.tensor(lp, device=self.device)
                                for lp in log_probs_shuf]
            # advantage/return are already Tensors
            adv_shuf = adv_shuf.to(self.device)
            ret_shuf = ret_shuf.to(self.device)

            # Update policy
            self.update_policy(vf_shuf, actions_shuf, log_probs_shuf_t, adv_shuf)
            # Update value
            self.update_value(gf_shuf, ret_shuf)

            avg_reward = np.mean(rewards)
            keys = info[0].keys()
    
            averages = {key: sum(d[key] for d in info) / len(info) for key in keys}

            self.logger.info(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}")
            print(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}", flush = True)
            print(averages, flush = True)

            # Save checkpoint
            if (iteration + 1) % self.cfg.save_model_interval == 0:
                checkpoint_path = f"ckpt_{self.cfg.name}_{iteration + 1}_{avg_reward:.2f}.pt"
                self.save_checkpoint(checkpoint_path)

            # Clear
            del states, actions, rewards, log_probs, dones

    def eval(self, num_samples, mean_action=True, info_list=None):
        """
        Runs multiple episodes with mean_action=True (greedy).
        """
        total_rewards = []
        all_plans = []
        for _ in range(num_samples):
            states, acts, rews, _, _ = self.collect_trajectory(mean_action=mean_action, info_list=info_list)
            total_rewards.append(np.sum(rews))
            all_plans.append(acts)
        avg_reward = np.mean(total_rewards)
        self.logger.info(f"Evaluation completed: Average Reward = {avg_reward:.2f}")
        return total_rewards, avg_reward, all_plans

    def infer(self, mean_action=True):
        """
        1-episode inference
        """
        infos = []
        rewards, avg_reward, plans = self.eval(num_samples=1, mean_action=mean_action, info_list=infos)
        plan = plans[0]
        total_reward = rewards[0]
        self.logger.info(f"Inference completed: Total Reward = {total_reward:.2f}")
        return plan, total_reward, infos
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import ceil

###############################################################################
# Rollout Buffer
###############################################################################
class RolloutBuffer:
    def __init__(self):
        """
        Stores transitions for PPO:
          obs: (grid_np, village_np)
          action: a Python structure describing the chosen villages & modes
          log_prob: float
          reward: float
          value_pred: float
          done: bool

        We do NOT store shapes or do partial unsqueezing. 
        The observation is kept as raw numpy arrays, so we can batch them later.
        """
        self.reset()

    def reset(self):
        self.ptr = 0
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.value_preds = []
        self.done = []

    def store(self, obs, action, log_prob, reward, value_pred, done):
        """
        obs: (grid_np, village_np) -> each is a numpy array
        """
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.value_preds.append(value_pred)
        self.done.append(done)
        self.ptr += 1

    def shuffle_data(self):
        """
        Randomly shuffle all transitions in unison, so we can do minibatch training.
        """
        idxs = np.arange(self.ptr)
        np.random.shuffle(idxs)

        self.obs        = [self.obs[i]        for i in idxs]
        self.actions    = [self.actions[i]    for i in idxs]
        self.log_probs  = [self.log_probs[i]  for i in idxs]
        self.rewards    = [self.rewards[i]    for i in idxs]
        self.value_preds= [self.value_preds[i]for i in idxs]
        self.done       = [self.done[i]       for i in idxs]


###############################################################################
# PPO Agent
###############################################################################
class PPOAgent:
    def __init__(
        self,
        policy_net,       # Your policy network (select villages + combos)
        value_net,        # Your value network (scalar value)
        state_encoder,    # state_encoder(grid_batch, village_batch)
        env,              # Environment with (reset, step)
        cfg               # Config object
    ):
        """
        We assume cfg has:
         - rollout_size (int)
         - gamma (float)
         - lam (float)
         - lr (float)
         - clip_range (float)
         - epochs (int)
         - mini_batch_size (int)
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.state_encoder = state_encoder
        self.env = env

        self.gamma = cfg.gamma
        self.lam = cfg.lam
        self.lr = cfg.lr
        self.clip_range = cfg.clip_range
        self.epochs = cfg.epochs
        self.mini_batch_size = cfg.mini_batch_size

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer()
        self.advantages = None
        self.returns = None

    def collect_trajectory(self, info_list = None):
        """
        Run the env for rollout_size steps, store transitions in buffer.
        Single env case => each step is one sample.
        """
        obs = self.env.reset()  # (grid_np, village_np)
        done = False

        for _ in range(self.rollout_size):
            grid_np, village_np = obs

            # 1) Convert to torch only for forward pass
            grid_tensor = torch.from_numpy(grid_np).float().unsqueeze(0)
            village_tensor = torch.from_numpy(village_np).float().unsqueeze(0)

            with torch.no_grad():
                # state_encoder -> (village_feats, global_feats)
                village_feats, global_feats = self.state_encoder(grid_tensor, village_tensor)
                actions_b, log_probs_b, _ = self.policy_net.select_action(village_feats, mean_action=False)
                action_for_env = actions_b[0]
                log_prob_action = log_probs_b[0].item()

                # value
                value_pred = self.value_net.forward(global_feats)  # shape [1]
                value_pred = value_pred.item()

            next_obs, reward, done, info = self.env.step(action_for_env)

            self.buffer.store(
                obs=obs, 
                action=action_for_env,
                log_prob=log_prob_action,
                reward=reward,
                value_pred=value_pred,
                done=done
            )

            obs = next_obs
            if done:
                obs = self.env.reset()
                done = False

    def compute_advantage(self, next_value=0.0):
        """
        Use GAE(lambda).
        If the last state is not terminal, pass next_value.
        """
        size = self.buffer.ptr
        advantages = [0.0]*size
        returns = [0.0]*size

        gae = 0.0
        for i in reversed(range(size)):
            if i == size-1:
                next_non_terminal = 1.0 - float(self.buffer.done[i])
                delta = self.buffer.rewards[i] + self.gamma * next_non_terminal * next_value - self.buffer.value_preds[i]
            else:
                next_non_terminal = 1.0 - float(self.buffer.done[i])
                delta = (self.buffer.rewards[i]
                         + self.gamma * next_non_terminal * self.buffer.value_preds[i+1]
                         - self.buffer.value_preds[i])

            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[i] = gae
            returns[i] = gae + self.buffer.value_preds[i]

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)

    def update(self):
        """
        Do PPO update with mini-batching and real batched forward passes:
         - Shuffle the buffer
         - For each mini-batch, create a stacked grid_batch, village_batch
         - Forward pass in one shot => new log_probs, new values
         - Compute PPO clipped loss
         - Optimize
        """
        buffer_size = self.buffer.ptr
        # Convert old arrays
        old_log_probs_all = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        advantages_all = self.advantages
        returns_all = self.returns
        value_preds_all = torch.tensor(self.buffer.value_preds, dtype=torch.float32)

        # Shuffle buffer
        self.buffer.shuffle_data()

        obs_all = self.buffer.obs
        actions_all = self.buffer.actions
        old_log_probs_all = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        adv_all = advantages_all
        ret_all = returns_all

        num_mini_batches = ceil(buffer_size / self.mini_batch_size)

        for epoch_i in range(self.epochs):
            mb_idx = 0
            for mb_i in range(num_mini_batches):
                start = mb_i * self.mini_batch_size
                end = min((mb_i+1)*self.mini_batch_size, buffer_size)

                batch_size = end - start

                # Extract slices
                obs_mb = obs_all[start:end]            # list of length batch_size
                actions_mb = actions_all[start:end]    # list of length batch_size
                old_log_probs_mb = old_log_probs_all[start:end]
                adv_mb = adv_all[start:end]
                ret_mb = ret_all[start:end]

                # 1) Build a single batched grid & village tensor
                #    We'll assume all grid_np have same shape, all village_np have same shape.
                grids_list = []
                villages_list = []
                for i_b in range(batch_size):
                    g_np, v_np = obs_mb[i_b]
                    grids_list.append(torch.from_numpy(g_np).float())     # shape e.g. [H, W, C]
                    villages_list.append(torch.from_numpy(v_np).float())  # shape e.g. [K, feats]

                # Stack => shape [batch_size, ...]
                grid_batch = torch.stack(grids_list, dim=0)       # e.g. [B, H, W, C] or [B, C, H, W]
                village_batch = torch.stack(villages_list, dim=0) # e.g. [B, K, feats]

                # 2) Encode in one shot
                village_feats, global_feats = self.state_encoder(grid_batch, village_batch)
                # village_feats => [B, N, embed_dim]
                # global_feats  => [B, global_dim]

                # 3) Recompute log_probs with the current policy in one pass
                new_log_probs_mb = self.policy_net.get_log_prob_entropy(village_feats, actions_mb)  

                # 4) Recompute value
                #    single pass => value_net.forward(global_feats) => shape [B]
                new_values_mb = self.value_net.forward(global_feats)

                # 5) ratio
                ratio = (new_log_probs_mb - old_log_probs_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # 6) value loss
                value_loss = (new_values_mb - ret_mb).pow(2).mean()

                # 7) final loss
                loss = policy_loss + 0.5*value_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

        # reset the buffer
        self.buffer.reset()

    def train_one_iteration(self):
        """
        1) collect_trajectory
        2) compute_advantage
        3) update
        """
        self.collect_trajectory()
        next_value = 0.0  # if not done at final step, compute from value net
        self.compute_advantage(next_value)
        self.update()

    def save_checkpoint(self, file_path):
        """
        Saves the model weights to a checkpoint file.

        Args:
        - file_path (str): Path to save the checkpoint.
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'state_encoder': self.state_encoder.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, file_path)
        self.logger.info(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        """
        Loads model weights from a checkpoint file.

        Args:
        - file_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(file_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.logger.info(f"Checkpoint loaded from {file_path}")
        
    def eval(self, num_samples, info_list=None):
        """
        Evaluates the agent by running the environment for a specified number of samples.

        Args:
        - num_samples (int): Number of episodes to run for evaluation.

        Returns:
        - dict: A dictionary containing evaluation metrics (e.g., average rewards).
        """
        total_rewards = []
        all_plans = []

        for _ in range(num_samples):
            states, masks, actions, rewards, _, _ = self.collect_trajectory(mean_action=True, info_list=info_list)
            total_rewards.append(np.sum(rewards))  # Total reward for this episode
            all_plans.append(actions)  # Store the plan (sequence of actions)

        avg_reward = np.mean(total_rewards)
        self.logger.info(f"Evaluation completed: Average Reward = {avg_reward:.2f}")

        return total_rewards, avg_reward, all_plans

    def infer(self):
        """
        Runs inference for a single episode and outputs the plan.

        Returns:
        - dict: A dictionary containing the plan (sequence of actions) and the total reward.
        """
        infos = []
        rewards, _, plans = self.eval(num_samples=1, info_list=infos)
        plan = plans[0]  # Extract the plan for the single episode
        total_reward = rewards[0]

        self.logger.info(f"Inference completed: Total Reward = {total_reward:.2f}")

        return plan, total_reward, infos
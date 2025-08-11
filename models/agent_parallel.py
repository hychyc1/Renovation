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
import multiprocessing
import concurrent.futures

class PPOAgentParallel:
    def __init__(self, env, cfg: Config, device, log_file="training.log"):
        """
        PPO Agent for training with the provided environment.
        Args:
            env: The environment with `reset` and `step` methods.
            cfg: Configuration object (hyperparameters, etc.).
            device: A torch.device (e.g. torch.device("cuda") or "cpu")
            log_file: Path to the log file for training logs.
        """
        # It is best to call this once in your main module
        # (place this in a if __name__=='__main__' block in your main script).
        multiprocessing.set_start_method('spawn', force=True)
        
        self.env = env
        self.cfg = cfg
        self.device = device  # e.g. torch.device("cuda") or "cpu"
        self.name = cfg.name

        # Initialize networks on the provided device.
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

        # Configuration for parallel sampling:
        if not hasattr(self.cfg, 'num_sample_workers'):
            self.cfg.num_sample_workers = 4  # default value
        if not hasattr(self.cfg, 'use_multiprocessing'):
            self.cfg.use_multiprocessing = True  # default to multiprocessing

        # Set up logging
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()
    
    @staticmethod
    def sample_worker_group(agent, mean_action, num_episodes, worker_id):
        """
        Static method to be executed in a subprocess.
        Each worker:
          1. Determines its CUDA device (cyclic assignment based on worker_id),
          2. Moves the agent's networks to that device,
          3. Collects `num_episodes` trajectories,
          4. Returns the aggregated trajectories.
        """
        # Determine available CUDA devices.
        available_devices = list(range(torch.cuda.device_count()))
        if not available_devices:
            device_id = 0  # fall back to CPU if no CUDA is available
        else:
            # Cyclic assignment: e.g. worker_id 0 -> cuda:0, worker_id 1 -> cuda:1, etc.
            device_id = available_devices[worker_id % len(available_devices)]
        torch.cuda.set_device(device_id)
        new_device = torch.device(f"cuda:{device_id}")
        
        # Move the networks to the new device (this is local to the worker process)
        agent.device = new_device
        agent.policy_net.to(new_device)
        agent.value_net.to(new_device)
        agent.state_encoder.to(new_device)
        
        all_states, all_actions, all_rewards, all_log_probs, all_dones = [], [], [], [], []
        for _ in tqdm(range(num_episodes)):
            traj = agent.collect_trajectory(mean_action=mean_action, info_list=None)
            states, actions, rewards, log_probs, dones = traj
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_dones.extend(dones)
        return all_states, all_actions, all_rewards, all_log_probs, all_dones

    def parse_state(self, state):
        """
        Converts the state into tensors on self.device.
        Args:
            state = (grid, villages, step)
        Returns:
            grid_tensor, village_tensor, year_tensor on self.device.
        """
        grid, villages, year = state
        grid_tensor = torch.stack(list(grid.values()), axis=0)
        village_tensor = torch.tensor(villages, dtype=torch.float32, device=self.device)
        village_tensor = village_tensor[:, :3]
        year_tensor = torch.tensor(year, device=self.device)
        return grid_tensor, village_tensor, year_tensor

    def collect_trajectory(self, mean_action=False, info_list=None):
        """
        Collects a single-episode trajectory by interacting with the environment.
        Returns:
            (states, actions, rewards, log_probs, dones)
        """
        states, actions, rewards, log_probs, dones = [], [], [], [], []
        state = self.env.reset()
        done = False
        while not done:
            grid_tensor, village_tensor, year_tensor = self.parse_state(state)
            with torch.no_grad():
                v_feats, g_feats = self.state_encoder(
                    grid_tensor.unsqueeze(0),
                    village_tensor.unsqueeze(0),
                    year_tensor.unsqueeze(0)
                )
                selected_actions, log_prob, _ = self.policy_net.select_action(v_feats, mean_action)
            action_for_env = selected_actions[0]
            next_state, reward, done, info = self.env.step(action_for_env)
            if info_list is not None:
                info_list.append(info)
            states.append((v_feats.squeeze(0), g_feats.squeeze(0)))
            actions.append(action_for_env)
            rewards.append(reward)
            log_probs.append(log_prob[0])
            dones.append(done)
            state = next_state
        return states, actions, rewards, log_probs, dones

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        returns = []
        next_value = 0
        advantage = 0
        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                next_value = 0
            delta = reward + self.gamma * next_value - value
            advantage = delta + self.gamma * (1 - done) * advantage
            returns.insert(0, reward + self.gamma * next_value)
            advantages.insert(0, advantage)
            next_value = value
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return advantages, returns

    def shuffle_trajectory(self, states, actions, log_probs, advantages, returns):
        n = len(states)
        idxs = torch.randperm(n, device=self.device)
        idxs_cpu = idxs.cpu().numpy()
        states_shuffled = [states[i] for i in idxs_cpu]
        actions_shuffled = [actions[i] for i in idxs_cpu]
        log_probs_shuffled = [log_probs[i] for i in idxs_cpu]
        advantages_shuffled = advantages[idxs]
        returns_shuffled = returns[idxs]
        return states_shuffled, actions_shuffled, log_probs_shuffled, advantages_shuffled, returns_shuffled

    def update_policy(self, village_features, actions, log_probs_old, advantages):
        log_probs_old = torch.stack(log_probs_old).to(self.device)
        T = len(village_features)
        for _ in range(self.num_epochs):
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                vf_batch = village_features[start:end]
                act_batch = actions[start:end]
                lpo_batch = log_probs_old[start:end]
                adv_batch = advantages[start:end]
                log_probs_new, entropy = self.policy_net.get_log_prob_entropy(vf_batch, act_batch)
                ratio = torch.exp(log_probs_new - lpo_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

    def update_value(self, global_features, returns):
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
            all_states, all_actions, all_rewards, all_log_probs, all_dones = [], [], [], [], []
            info = []  # aggregate environment info if needed

            # --- Parallel trajectory sampling ---
            # Use a fixed number of worker processes.
            num_workers = self.cfg.num_sample_workers
            # Split the total number of episodes equally (handle remainder)
            base = self.num_episodes_per_iteration // num_workers
            remainder = self.num_episodes_per_iteration % num_workers
            worker_episode_counts = [base + (1 if i < remainder else 0) for i in range(num_workers)]

            Executor = (concurrent.futures.ProcessPoolExecutor
                        if self.cfg.use_multiprocessing
                        else concurrent.futures.ThreadPoolExecutor)
            print(f"Collecting {self.num_episodes_per_iteration} episodes using {num_workers} workers...", flush=True)
            with Executor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(PPOAgentParallel.sample_worker_group, self, False, count, i)
                    for i, count in enumerate(worker_episode_counts)
                ]
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(futures), desc="Sampling episodes"):
                    states_i, actions_i, rewards_i, log_probs_i, dones_i = future.result()
                    all_states.extend(states_i)
                    all_actions.extend(actions_i)
                    all_rewards.extend(rewards_i)
                    all_log_probs.extend(log_probs_i)
                    all_dones.extend(dones_i)
            print("Finished collecting episodes.", flush=True)
            # --- End of Parallel Sampling ---

            # Process collected trajectories.
            _, global_feats_list = zip(*all_states)
            global_feats_tensor = torch.stack(global_feats_list).to(self.device)
            with torch.no_grad():
                values_t = self.value_net(global_feats_tensor).squeeze(-1)
            values_np = values_t.cpu().numpy()
            advantages, returns = self.compute_advantages(all_rewards, values_np, all_dones)
            states_shuf, actions_shuf, log_probs_shuf, adv_shuf, ret_shuf = self.shuffle_trajectory(
                all_states, all_actions, all_log_probs, advantages, returns
            )
            vf_shuf_list, gf_shuf_list = zip(*states_shuf)
            vf_shuf = torch.stack(vf_shuf_list).to(self.device)
            gf_shuf = torch.stack(gf_shuf_list).to(self.device)
            log_probs_shuf_t = [lp if torch.is_tensor(lp) else torch.tensor(lp, device=self.device)
                                for lp in log_probs_shuf]
            adv_shuf = adv_shuf.to(self.device)
            ret_shuf = ret_shuf.to(self.device)

            self.update_policy(vf_shuf, actions_shuf, log_probs_shuf_t, adv_shuf)
            self.update_value(gf_shuf, ret_shuf)

            avg_reward = np.mean(all_rewards)
            if len(info) > 0:
                keys = info[0].keys()
                averages = {key: sum(d[key] for d in info) / len(info) for key in keys}
                print(averages, flush=True)

            self.logger.info(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}")
            print(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}", flush=True)

            if (iteration + 1) % self.cfg.save_model_interval == 0:
                checkpoint_path = f"ckpt_{self.cfg.name}_{iteration + 1}_{avg_reward:.2f}.pt"
                self.save_checkpoint(checkpoint_path)
            # Clean up memory between iterations.
            del all_states, all_actions, all_rewards, all_log_probs, all_dones

    def eval(self, num_samples, mean_action=True, info_list=None):
        total_rewards = []
        all_plans = []
        for _ in range(num_samples):
            traj = self.collect_trajectory(mean_action=mean_action, info_list=info_list)
            states, acts, rews, _, _ = traj
            total_rewards.append(np.sum(rews))
            all_plans.append(acts)
        avg_reward = np.mean(total_rewards)
        self.logger.info(f"Evaluation completed: Average Reward = {avg_reward:.2f}")
        return total_rewards, avg_reward, all_plans

    def infer(self, mean_action=True):
        infos = []
        rewards, avg_reward, plans = self.eval(num_samples=1, mean_action=mean_action, info_list=infos)
        plan = plans[0]
        total_reward = rewards[0]
        self.logger.info(f"Inference completed: Total Reward = {total_reward:.2f}")
        return plan, total_reward, infos

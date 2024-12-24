import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from models.policy import PolicyHead
from models.state_encoder import CNNStateEncoder
from models.value import ValueNetwork
from Config.config import Config


class PPOAgent:
    def __init__(self, cfg: Config, env, log_file="training.log"):
        """
        PPO Agent for training with the provided environment.

        Args:
        - cfg (object): Configuration object containing hyperparameters and network configurations.
        - env: The environment with `reset` and `step` methods.
        - log_file (str): Path to the log file for training logs.
        """
        self.env = env
        self.cfg = cfg

        # Initialize networks
        self.policy_net = PolicyHead(cfg)
        self.value_net = ValueNetwork(cfg)
        self.state_encoder = CNNStateEncoder(cfg)

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
        Converts a dictionary of 2D arrays (state) into a 3D tensor.

        Args:
        - state (dict): Dictionary with k keys, where each value is a 2D array of shape (n, m).

        Returns:
        - torch.Tensor: A tensor of shape (batch_size, k, n, m).
        """
        state_tensor = np.stack(list(state.values()), axis=0)
        return torch.tensor(state_tensor, dtype=torch.float32)

    def collect_trajectory(self, mean_action=False, info_list=None):
        """
        Collects a trajectory by interacting with the environment.

        Args:
        - mean_action (bool): If True, selects deterministic actions.
        - info_list (list): Optional list to store additional environment info.

        Returns:
        - states (list): List of encoded states.
        - masks (list): List of masks for valid grids (AREA > 0).
        - actions (list): List of actions taken as (i, j, combination, strength).
        - rewards (list): List of rewards received.
        - log_probs (list): Log probabilities of actions taken.
        - dones (list): List of done flags.
        """
        states, masks, actions, rewards, log_probs, dones = [], [], [], [], [], []

        state = self.env.reset()
        done = False

        while not done:
            # Compute mask for valid grids
            mask = torch.tensor(state['AREA'] > 0, dtype=torch.bool)  # Mask as a boolean tensor

            # Convert state to tensor and ensure it is batched
            state_tensor = self.parse_state(state).unsqueeze(0)  # Add batch dimension
            state_encoded = self.state_encoder(state_tensor).squeeze(0)  # Remove batch dimension after encoding

            with torch.no_grad():
                selected_actions = self.policy_net.select_action(state_encoded, mask, mean_action)
                log_prob, _ = self.policy_net.get_log_prob_entropy(state_encoded, selected_actions, mask)

            next_state, reward, done, info = self.env.step(selected_actions[0])

            if info_list is not None:
                info_list.append(info)

            # Record trajectory
            states.append(state_encoded)
            masks.append(mask)  # Store mask for valid grids
            actions.append(selected_actions[0])
            rewards.append(reward)
            log_probs.append(log_prob[0])
            dones.append(done)

            state = next_state

        return states, masks, actions, rewards, log_probs, dones
    
    def compute_advantages(self, rewards, values, dones):
        """
        Computes advantages using Generalized Advantage Estimation (GAE).

        Args:
        - rewards (list): List of rewards.
        - values (list): List of state values.
        - dones (list): List of done flags.

        Returns:
        - advantages (torch.Tensor): Computed advantages.
        - returns (torch.Tensor): Discounted returns.
        """
        advantages = []
        returns = []
        next_value = 0
        advantage = 0

        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            next_value = 0 if done else next_value
            delta = reward + self.gamma * next_value - value
            advantage = delta + self.gamma * (1 - done) * advantage
            returns.insert(0, reward + self.gamma * next_value)
            advantages.insert(0, advantage)
            next_value = value

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def shuffle_trajectory(self, states, masks, actions, log_probs, advantages, returns):
        """
        Permutes the trajectory to reduce variance during training.

        Args:
        - states (list): States from the trajectory.
        - actions (list): Actions from the trajectory.
        - log_probs (list): Log probabilities of the actions.
        - advantages (torch.Tensor): Advantages computed from the trajectory.
        - returns (torch.Tensor): Discounted returns.

        Returns:
        - Shuffled versions of all inputs.
        """
        indices = torch.randperm(len(states))
        states = [states[i] for i in indices]
        masks = [masks[i] for i in indices]
        actions = [actions[i] for i in indices]
        log_probs = [log_probs[i] for i in indices]
        advantages = advantages[indices]
        returns = returns[indices]
        return states, masks, actions, log_probs, advantages, returns

    def update_policy(self, states, masks, actions, log_probs_old, advantages):
        """
        Updates the policy network using PPO's clipped surrogate objective.

        Args:
        - states (list): Encoded states.
        - actions (list): Actions taken corresponding to states.
        - log_probs_old (list): Log probabilities under the old policy.
        - advantages (torch.Tensor): Computed advantages (one per step, shared across all actions).
        """
        for _ in range(self.num_epochs):
            # print(f'AA {len(states)} {self.batch_size} BB')
            for idx in tqdm(range(0, len(states), self.batch_size)):
                # Ensure the batch doesn't exceed dataset length
                end_idx = min(idx + self.batch_size, len(states))

                # Get mini-batch
                state_batch = torch.stack(states[idx:end_idx])  # Batch of states
                action_batch = actions[idx:end_idx]  # Batch of actions
                log_probs_old_batch = torch.stack(log_probs_old[idx:end_idx])  # Batch of log_probs
                advantages_batch = advantages[idx:end_idx]  # Batch of advantages

                # Compute new log probabilities and entropy
                log_probs_new, entropy = self.policy_net.get_log_prob_entropy(state_batch, action_batch, masks)

                # Expand the single advantage per step to match the action dimension
                advantages_expanded = advantages_batch.unsqueeze(-1).expand_as(log_probs_new)

                # Compute ratios
                ratios = torch.exp(log_probs_new - log_probs_old_batch)

                # PPO objective
                surrogate1 = ratios * advantages_expanded
                surrogate2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages_expanded
                policy_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coef * entropy.mean()

                # Backpropagation
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True) 
                self.policy_optimizer.step()

    def update_value(self, states, returns):
        """
        Updates the value network using mean squared error loss.

        Args:
        - states (list): Encoded states.
        - returns (torch.Tensor): Target returns.
        """
        for _ in range(self.num_epochs):
            for idx in range(0, len(states), self.batch_size):
                # Ensure the batch doesn't exceed dataset length
                end_idx = min(idx + self.batch_size, len(states))

                # Get mini-batch
                state_batch = torch.stack(states[idx:end_idx])
                return_batch = returns[idx:end_idx]

                # Compute value predictions
                values = self.value_net(state_batch).squeeze(-1)
                value_loss = self.value_pred_coef * (values - return_batch).pow(2).mean()

                # Backpropagation
                self.value_optimizer.zero_grad()
                value_loss.backward(retain_graph=True)  
                self.value_optimizer.step()

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
        
    def train(self, num_iterations):
        """
        Trains the agent for a given number of iterations.

        Args:
        - num_iterations (int): Number of training iterations.
        """
        for iteration in range(num_iterations):
            states, masks, actions, rewards, log_probs, dones = [], [], [], [], [], []

            for _ in tqdm(range(self.num_episodes_per_iteration)):
                cur_states, cur_masks, cur_actions, cur_rewards, cur_log_probs, cur_dones = self.collect_trajectory()
                states.extend(cur_states)
                masks.extend(cur_masks)
                actions.extend(cur_actions)
                rewards.extend(cur_rewards)
                log_probs.extend(cur_log_probs)
                dones.extend(cur_dones)

            # Compute values and advantages
            states_tensor = torch.stack(states)
            values = self.value_net(states_tensor).squeeze(-1).detach().numpy()
            advantages, returns = self.compute_advantages(rewards, values, dones)

            # Shuffle the trajectory
            states, masks, actions, log_probs, advantages, returns = self.shuffle_trajectory(
                states, masks, actions, log_probs, advantages, returns
            )

            # Update policy and value networks
            self.update_policy(states, masks, actions, log_probs, advantages)
            self.update_value(states, returns)

            # Log average reward
            avg_reward = np.mean(rewards)
            self.logger.info(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}")
            tqdm.write(f"Iteration {iteration + 1}/{num_iterations} - Average Reward: {avg_reward}")

            # Save checkpoint every 20 iterations
            if (iteration + 1) % self.cfg.save_model_interval == 0:
                checkpoint_path = f"checkpoint_iter_{iteration + 1}_reward_{avg_reward: .2f}.pt"
                self.save_checkpoint(checkpoint_path)

            # Clear trajectory data to save memory
            del states, masks, actions, rewards, log_probs, dones

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

        for _ in tqdm(range(num_samples), desc="Evaluating"):
            states, masks, actions, rewards, _, _ = self.collect_trajectory(mean_action=True, info_list=info_list)
            total_rewards.append(np.sum(rewards))  # Total reward for this episode
            all_plans.append(actions)  # Store the plan (sequence of actions)

        avg_reward = np.mean(total_rewards)
        self.logger.info(f"Evaluation completed: Average Reward = {avg_reward:.2f}")
        tqdm.write(f"Evaluation completed: Average Reward = {avg_reward:.2f}")

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

        tqdm.write(f"Inference completed: Total Reward = {total_reward:.2f}")
        self.logger.info(f"Inference completed: Total Reward = {total_reward:.2f}")

        return plan, total_reward, infos

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.policy import PolicyHead
from models.state_encoder import CNNStateEncoder
from models.value import ValueNetwork

class PPOAgent:
    def __init__(self, cfg, env):
        """
        PPO Agent for training with the provided environment.

        Args:
        - cfg (object): Configuration object containing hyperparameters and network configurations.
        - env: The environment with `reset` and `step` methods.
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
        self.num_episodes_per_iteration = cfg.num_episodes_per_iteration  # Number of episodes per iteration

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
        print(f"Checkpoint saved to {file_path}.")

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
        print(f"Checkpoint loaded from {file_path}.")

    def parse_state(self, state):
        """
        Converts a dictionary of 2D arrays (state) into a 3D tensor.

        Args:
        - state (dict): Dictionary with k keys, where each value is a 2D array of shape (n, m).

        Returns:
        - torch.Tensor: A tensor of shape (k, n, m).
        """
        state_tensor = np.stack(list(state.values()), axis=0)
        return torch.tensor(state_tensor, dtype=torch.float32)

    def collect_trajectory(self):
        """
        Collects a single trajectory by interacting with the environment.

        Returns:
        - states (list): List of encoded states.
        - actions (list): List of actions taken as (i, j, combination, strength).
        - rewards (list): List of rewards received.
        - log_probs (list): Log probabilities of actions taken.
        - dones (list): List of done flags.
        """
        states, actions, rewards, log_probs, dones = [], [], [], [], []

        state = self.env.reset()
        done = False

        while not done:
            # Convert state to tensor and encode
            state_tensor = self.parse_state(state).unsqueeze(0)  # Add batch dimension
            state_encoded = self.state_encoder(state_tensor).squeeze(0)  # Remove batch dimension after encoding

            # Select action
            with torch.no_grad():
                selected_actions = self.policy_net.select_action(state_encoded)
                log_prob, _ = self.policy_net.get_log_prob_entropy(state_encoded, selected_actions)

            # Interact with the environment
            next_state, reward, done, _ = self.env.step(selected_actions[0])

            # Record trajectory
            states.append(state_encoded)  # Do not detach; needed for gradient computation
            actions.append(selected_actions[0])
            rewards.append(reward)
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state

        return states, actions, rewards, log_probs, dones

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

    def train(self, num_iterations):
        """
        Trains the agent for a given number of iterations.

        Args:
        - num_iterations (int): Number of training iterations.
        """
        for iteration in tqdm(range(num_iterations)):
            all_states, all_actions, all_rewards, all_log_probs, all_dones = [], [], [], [], []

            # Collect multiple trajectories
            for _ in tqdm(range(self.num_episodes_per_iteration)):
                states, actions, rewards, log_probs, dones = self.collect_trajectory()
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_log_probs.extend(log_probs)
                all_dones.extend(dones)

            # Compute value estimates and advantages
            values = [self.value_net(state).item() for state in all_states]
            advantages, returns = self.compute_advantages(all_rewards, values, all_dones)

            # Shuffle the trajectory
            all_states, all_actions, all_log_probs, advantages, returns = self.shuffle_trajectory(
                all_states, all_actions, all_log_probs, advantages, returns
            )

            # Update policy and value networks
            self.update_policy(all_states, all_actions, all_log_probs, advantages)
            self.update_value(all_states, returns)

            print(f"Iteration {iteration + 1}/{num_iterations} complete.")
            print(f"rewards {np.average(all_rewards)}")

            # Clear trajectory data
            del all_states, all_actions, all_rewards, all_log_probs, all_dones


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.policy import PolicyHead
from models.state_encoder import CNNStateEncoder
from models.value import ValueNetwork
import numpy as np

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

    def parse_state(self, state):
        """
        Converts a dictionary of 2D arrays (state) into a 3D tensor.

        Args:
        - state (dict): Dictionary with k keys, where each value is a 2D array of shape (n, m).

        Returns:
        - torch.Tensor: A tensor of shape (n, m, k).
        """
        state_tensor = np.stack(list(state.values()), axis=0)
        state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
        print(state_tensor.shape)
        return state_tensor

    def collect_trajectory(self):
        """
        Collects a trajectory by interacting with the environment.

        Returns:
        - states (list): List of states encountered.
        - actions (list): List of actions taken as (i, j, combination, strength).
        - rewards (list): List of rewards received.
        - log_probs (list): Log probabilities of actions taken.
        - dones (list): List of done flags.
        """
        states, actions, rewards, log_probs, dones = [], [], [], [], []

        state = self.env.reset()
        done = False

        while not done:
            # Convert state to tensor
            state_tensor = self.state_encoder.forward([self.parse_state(state)])
            print(state_tensor.shape)
            # Select action
            with torch.no_grad():
                # action_probs = self.policy_net(state_tensor)
                selected_actions = self.policy_net.select_action(state_tensor)
                log_prob, _ = self.policy_net.get_log_prob_entropy(state_tensor, [selected_actions])

            # Interact with the environment
            next_state, reward, done, _ = self.env.step(selected_actions[0])  # Unbatch the action for the env

            # Record trajectory
            states.append(state)
            actions.append(selected_actions[0])  # Store unbatched action
            rewards.append(reward)
            log_probs.append(log_prob.squeeze(0))  # Unbatch the log prob
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

    def shuffle_trajectory(self, states, actions, log_probs, advantages, returns):
        """
        Permutes the trajectory to reduce variance during training.

        Args:
        - states (torch.Tensor): States from the trajectory.
        - actions (list): Actions from the trajectory.
        - log_probs (torch.Tensor): Log probabilities of the actions.
        - advantages (torch.Tensor): Advantages computed from the trajectory.
        - returns (torch.Tensor): Discounted returns.

        Returns:
        - Shuffled versions of all inputs.
        """
        indices = torch.randperm(len(states))
        states = states[indices]
        actions = [actions[i] for i in indices]
        log_probs = log_probs[indices]
        advantages = advantages[indices]
        returns = returns[indices]
        return states, actions, log_probs, advantages, returns

    def update_policy(self, states, actions, log_probs_old, advantages):
        """
        Updates the policy network using PPO's clipped surrogate objective.

        Args:
        - states (torch.Tensor): Batch of states.
        - actions (list): List of actions taken as (i, j, combination, strength).
        - log_probs_old (torch.Tensor): Log probabilities of actions under the old policy.
        - advantages (torch.Tensor): Computed advantages.
        """
        for _ in range(self.num_epochs):
            for idx in range(0, len(states), self.batch_size):
                # Mini-batch sampling
                state_batch = states[idx:idx + self.batch_size]
                action_batch = actions[idx:idx + self.batch_size]
                log_probs_old_batch = log_probs_old[idx:idx + self.batch_size]
                advantages_batch = advantages[idx:idx + self.batch_size]

                # Compute new log probabilities and entropy
                log_probs_new, entropy = self.policy_net.get_log_prob_entropy(state_batch, action_batch)

                # Ratio of probabilities
                ratios = torch.exp(log_probs_new - log_probs_old_batch)

                # Clipped surrogate objective
                surrogate1 = ratios * advantages_batch
                surrogate2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coef * entropy.mean()

                # Backpropagate
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

    def update_value(self, states, returns):
        """
        Updates the value network using mean squared error loss.

        Args:
        - states (torch.Tensor): Batch of states.
        - returns (torch.Tensor): Target returns.
        """
        for _ in range(self.num_epochs):
            for idx in range(0, len(states), self.batch_size):
                # Mini-batch sampling
                state_batch = states[idx:idx + self.batch_size]
                return_batch = returns[idx:idx + self.batch_size]

                # Compute value loss
                values = self.value_net(state_batch).squeeze(-1)
                value_loss = self.value_pred_coef * (values - return_batch).pow(2).mean()

                # Backpropagate
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def train(self, num_iterations):
        """
        Trains the agent for a given number of iterations.

        Args:
        - num_iterations (int): Number of training iterations.
        """
        for iteration in range(num_iterations):
            # Collect trajectory
            states, actions, rewards, log_probs, dones = self.collect_trajectory()

            # Compute values and advantages
            states_tensor = torch.tensor(states, dtype=torch.float32)
            values = self.value_net(states_tensor).squeeze(-1).detach().numpy()
            advantages, returns = self.compute_advantages(rewards, values, dones)

            # Shuffle the trajectory
            states_tensor, actions, log_probs, advantages, returns = self.shuffle_trajectory(
                states_tensor, actions, log_probs, advantages, returns
            )

            # Update networks
            self.update_policy(states_tensor, actions, log_probs, advantages)
            self.update_value(states_tensor, returns)

            print(f"Iteration {iteration + 1}/{num_iterations} complete.")
            print(f"Rewards Collected {np.average(rewards)}")

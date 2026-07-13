from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    PPO uses an actor to output action probabilities and a critic to estimate V(s).
    For discrete CartPole actions, the actor outputs logits for [left, right].
    """
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        # Shared feature extractor: both actor and critic read the same state features.
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        # Actor head: produces unnormalized action scores. Categorical(logits=...)
        # will turn these into action probabilities internally.
        self.actor = nn.Linear(hidden, action_dim)
        # Critic head: predicts V(s), the expected return from this state.
        self.critic = nn.Linear(hidden, 1)

    def forward(self, states):
        features = self.shared(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values


@dataclass
class PPOMetrics:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


class RolloutBuffer:
    """Short-lived on-policy storage.

    PPO collects a fresh rollout with the current policy, updates on that rollout
    for a few epochs, then discards it. This is intentionally not a DQN-style
    replay buffer.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, state, action, log_prob, reward, done, value):
        # log_prob and value must be stored at collection time. During update,
        # log_prob represents pi_old(a|s), and value is used to compute advantage.
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        batch_size=64,
        entropy_coef=0.001,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden=64,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCritic(state_dim, action_dim, hidden=hidden)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def act(self, state, greedy=False):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(state_t)
            dist = Categorical(logits=logits)

            if greedy:
                # Evaluation mode: choose the most likely action.
                action = torch.argmax(logits, dim=1)
            else:
                # Training mode: sample so the policy can keep exploring.
                action = dist.sample()

            # Store log pi_old(a|s). PPO compares this old probability with the
            # new probability after the network has been updated.
            log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def value(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(state_t)
        return float(value.item())

    def evaluate_actions(self, states, actions):
        # Recompute log pi_new(a|s), entropy, and V(s) for a batch of old actions.
        # This is where "new policy vs old policy" is measured.
        logits, values = self.network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def compute_returns_and_advantages(self, next_value):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0

        # Generalized Advantage Estimation (GAE).
        # delta is a one-step TD error. The recursive gae line mixes many-step
        # TD errors, controlled by gae_lambda.
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae

        # Critic target: return = advantage + baseline V(s).
        returns = advantages + np.array(self.buffer.values, dtype=np.float32)
        return returns, advantages

    def update(self, next_value=0.0):
        returns, advantages = self.compute_returns_and_advantages(next_value)

        # Convert the rollout into tensors once. These are on-policy samples
        # collected by the previous policy pi_old.
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)

        # Normalizing advantages usually makes policy-gradient updates less noisy.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        last_metrics = None
        num_samples = len(states)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                log_probs, entropy, values = self.evaluate_actions(
                    states[batch_idx],
                    actions[batch_idx],
                )

                # ratio = pi_new(a|s) / pi_old(a|s).
                # We stored log pi_old during rollout collection; log_probs is
                # log pi_new under the current network.
                ratios = torch.exp(log_probs - old_log_probs[batch_idx])

                # Unclipped policy-gradient objective. If advantage is positive,
                # increasing the action probability helps; if negative, decreasing
                # it helps.
                unclipped = ratios * advantages_t[batch_idx]

                # PPO's main safety device: do not let the new policy probability
                # move too far from the old policy probability in one update.
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages_t[batch_idx]

                # Negative because PyTorch optimizers minimize losses, while PPO
                # wants to maximize the clipped policy objective.
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Critic learns to predict the rollout return target.
                value_loss = nn.functional.mse_loss(values, returns_t[batch_idx])

                # Entropy bonus discourages the policy from becoming deterministic
                # too early.
                entropy_bonus = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping is another stability guard.
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Diagnostics: large KL or high clip fraction means the policy
                    # is changing aggressively.
                    approx_kl = (old_log_probs[batch_idx] - log_probs).mean()
                    clip_fraction = (
                        (ratios - 1.0).abs() > self.clip_epsilon
                    ).float().mean()

                last_metrics = PPOMetrics(
                    loss=float(loss.item()),
                    policy_loss=float(policy_loss.item()),
                    value_loss=float(value_loss.item()),
                    entropy=float(entropy_bonus.item()),
                    approx_kl=float(approx_kl.item()),
                    clip_fraction=float(clip_fraction.item()),
                )

        self.buffer.clear()
        return last_metrics

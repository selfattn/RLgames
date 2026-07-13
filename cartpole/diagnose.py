import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartpole.agent import QNetwork, ReplayBuffer
from cartpole.env import CartPoleEnv


@dataclass
class ExperimentConfig:
    name: str
    note: str
    use_replay: bool = True
    use_target_network: bool = True
    epsilon_decay: float = 0.995
    lr: float = 1e-3
    seed_offset: int = 0


class DiagnosticDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        *,
        use_replay=True,
        use_target_network=True,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=20,
    ):
        self.action_dim = action_dim
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

        if self.use_replay:
            if len(self.replay_buffer) < self.batch_size:
                return None
            batch = self.replay_buffer.sample(self.batch_size)
        else:
            batch = self._transition_to_batch(state, action, reward, next_state, done)

        return self._learn_from_batch(batch)

    def _transition_to_batch(self, state, action, reward, next_state, done):
        return (
            torch.tensor(np.array([state]), dtype=torch.float32),
            torch.tensor([action], dtype=torch.long),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(np.array([next_state]), dtype=torch.float32),
            torch.tensor([done], dtype=torch.float32),
        )

    def _learn_from_batch(self, batch):
        states, actions, rewards, next_states, dones = batch

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_net = self.target_net if self.use_target_network else self.q_net
        with torch.no_grad():
            next_q_values = next_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.use_target_network and self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train_experiment(config, episodes=350, max_steps=500, base_seed=42):
    set_seed(base_seed + config.seed_offset)
    env = CartPoleEnv()
    agent = DiagnosticDQNAgent(
        env.state_dim,
        env.action_dim,
        use_replay=config.use_replay,
        use_target_network=config.use_target_network,
        lr=config.lr,
        epsilon_decay=config.epsilon_decay,
    )

    rewards = []
    losses = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            loss = agent.observe(state, action, reward, next_state, done)

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        agent.decay_epsilon()

    eval_rewards = evaluate(agent, episodes=8, max_steps=max_steps)
    return {
        "config": config,
        "agent": agent,
        "rewards": np.array(rewards, dtype=float),
        "losses": np.array(losses, dtype=float),
        "eval_rewards": np.array(eval_rewards, dtype=float),
        "max_steps": max_steps,
    }


def evaluate(agent, episodes=8, max_steps=300):
    env = CartPoleEnv()
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action = agent.act(state, greedy=True)
            state, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return rewards


def summarize(result):
    rewards = result["rewards"]
    losses = result["losses"]
    eval_rewards = result["eval_rewards"]
    agent = result["agent"]
    max_steps = result["max_steps"]

    first_20 = rewards[:20]
    last_20 = rewards[-20:]
    recent_losses = losses[-200:] if len(losses) else np.array([np.nan])

    return {
        "train_first20": float(np.mean(first_20)),
        "train_last20": float(np.mean(last_20)),
        "train_last20_std": float(np.std(last_20)),
        "train_max": float(np.max(rewards)),
        "eval_avg": float(np.mean(eval_rewards)),
        "eval_min": float(np.min(eval_rewards)),
        "loss_last": float(np.nanmean(recent_losses)),
        "updates": agent.update_count,
        "epsilon": agent.epsilon,
        "max_steps": max_steps,
    }


def print_report(results):
    print("CartPole DQN diagnostic report")
    print("Same environment, same network size. Each row removes or changes one stabilizer.")
    print()
    print(
        f"{'Experiment':<20} {'Train first20':>13} {'Train last20':>12} "
        f"{'Train max':>10} {'Eval avg':>9} {'Eval min':>9} "
        f"{'Std':>8} {'Eps':>7} {'Loss':>11}"
    )
    print("-" * 108)

    summaries = []
    for result in results:
        summary = summarize(result)
        summaries.append(summary)
        cfg = result["config"]
        print(
            f"{cfg.name:<20} "
            f"{summary['train_first20']:13.1f} "
            f"{summary['train_last20']:12.1f} "
            f"{summary['train_max']:10.1f} "
            f"{summary['eval_avg']:9.1f} "
            f"{summary['eval_min']:9.1f} "
            f"{summary['train_last20_std']:8.1f} "
            f"{summary['epsilon']:7.3f} "
            f"{summary['loss_last']:11.1f}"
        )

    print("\nReading guide:")
    for result, summary in zip(results, summaries):
        cfg = result["config"]
        max_steps = summary["max_steps"]
        print(f"- {cfg.name}: {cfg.note}")
        if summary["eval_avg"] >= 0.75 * max_steps:
            print("  Symptom: greedy evaluation is strong in this run; still check the ablations before trusting the code.")
        elif summary["train_max"] >= 0.75 * max_steps and summary["eval_avg"] < 0.35 * max_steps:
            print("  Symptom: training had high-reward spikes, but the greedy policy did not reliably keep them.")
        elif summary["loss_last"] > 10000:
            print("  Symptom: loss exploded; inspect bootstrapping target and target-network usage.")
        elif summary["eval_avg"] < 0.15 * max_steps:
            print("  Symptom: greedy evaluation is still near random; inspect exploration and target calculation.")
        elif summary["train_last20_std"] > 0.18 * max_steps:
            print("  Symptom: reward variance is high; the training loop may be unstable even if max reward looks good.")
        else:
            print("  Symptom: no obvious failure in this short run; compare against the ablations before trusting it.")


if __name__ == "__main__":
    experiments = [
        ExperimentConfig(
            name="normal_dqn",
            note="Replay buffer + target network. This is the reference, not a proof of correctness.",
            seed_offset=0,
        ),
        ExperimentConfig(
            name="no_replay",
            note="Learns from only the latest transition. Watch for noisy updates and poor generalization.",
            use_replay=False,
            seed_offset=100,
        ),
        ExperimentConfig(
            name="no_target",
            note="Bootstraps from the online network itself. Watch for target drift and high variance.",
            use_target_network=False,
            seed_offset=200,
        ),
        ExperimentConfig(
            name="fast_epsilon_decay",
            note="Exploration drops too quickly. A run can look decisive while converging to a weak policy.",
            epsilon_decay=0.80,
            seed_offset=300,
        ),
    ]

    results = [train_experiment(config) for config in experiments]
    print_report(results)

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartpole.env import CartPoleEnv
from cartpole.ppo_agent import PPOAgent


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def evaluate(agent, episodes=8, max_steps=500):
    env = CartPoleEnv()
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action, _, _ = agent.act(state, greedy=True)
            state, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return rewards


def train(
    updates=80,
    steps_per_update=1024,
    max_steps=500,
    eval_every=10,
    seed=42,
):
    set_seed(seed)
    env = CartPoleEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)

    state = env.reset()
    episode_reward = 0.0
    episode_len = 0
    episode_rewards = []

    print("Starting PPO Training on CartPole...")
    print("PPO is on-policy: collect fresh rollouts, then update the actor-critic with a clipped objective.\n")

    for update in range(1, updates + 1):
        rollout_rewards = []

        for _ in range(steps_per_update):
            action, log_prob, value = agent.act(state)
            next_state, reward, done = env.step(action)

            episode_reward += reward
            episode_len += 1

            timeout = episode_len >= max_steps
            terminal = done or timeout
            agent.buffer.push(state, action, log_prob, reward, terminal, value)

            state = next_state

            if terminal:
                episode_rewards.append(episode_reward)
                rollout_rewards.append(episode_reward)
                state = env.reset()
                episode_reward = 0.0
                episode_len = 0

        next_value = 0.0 if episode_len == 0 else agent.value(state)
        metrics = agent.update(next_value=next_value)

        recent = episode_rewards[-20:] if episode_rewards else [0.0]
        line = (
            f"Update {update:3d}/{updates} | "
            f"AvgReward20 {np.mean(recent):7.1f} | "
            f"RolloutEpisodes {len(rollout_rewards):3d} | "
            f"Loss {metrics.loss:8.3f} | "
            f"KL {metrics.approx_kl:7.4f} | "
            f"ClipFrac {metrics.clip_fraction:5.2f}"
        )

        if update % eval_every == 0 or update == 1 or update == updates:
            eval_rewards = evaluate(agent, episodes=5, max_steps=max_steps)
            line += f" | EvalAvg {np.mean(eval_rewards):7.1f} | EvalMin {np.min(eval_rewards):5.1f}"

        print(line)

    return agent, episode_rewards


def demonstrate(agent, max_steps=500, delay=0.03):
    env = CartPoleEnv()
    state = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        clear_screen()
        print(f"PPO GREEDY DEMONSTRATION | Step {step} | Reward: {total_reward:.0f}")
        print("Actor chooses argmax action probability.\n")
        print(env.render(width=50))

        action, _, _ = agent.act(state, greedy=True)
        state, reward, done = env.step(action)
        total_reward += reward
        time.sleep(delay)

        if done:
            break

    clear_screen()
    print(f"Final PPO demonstration reward: {int(total_reward)}")
    print(env.render(width=50))


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on the hand-written CartPole environment.")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--steps-per-update", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Show an ASCII greedy-policy demo after training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent, rewards = train(
        updates=args.updates,
        steps_per_update=args.steps_per_update,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        seed=args.seed,
    )

    final_eval = evaluate(agent, episodes=8, max_steps=args.max_steps)
    print("\nFinal PPO evaluation")
    print(f"Average reward: {np.mean(final_eval):.1f}")
    print(f"Minimum reward: {np.min(final_eval):.1f}")
    print(f"Max training reward: {max(rewards) if rewards else 0:.1f}")

    if args.demo:
        demonstrate(agent, max_steps=args.max_steps)

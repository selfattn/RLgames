import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cliff_walking.agent import QLearningAgent, SARSAAgent
from cliff_walking.env import CliffWalkingEnv


def train(agent_type, episodes=800, max_steps=500, seed=42):
    np.random.seed(seed)
    env = CliffWalkingEnv()
    if agent_type == "q_learning":
        agent = QLearningAgent(env, alpha=0.5, gamma=1.0, epsilon=0.1)
    elif agent_type == "sarsa":
        agent = SARSAAgent(env, alpha=0.5, gamma=1.0, epsilon=0.1)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    rewards = []
    cliff_falls = []

    for _ in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        falls = 0

        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)

            if agent_type == "q_learning":
                agent.update(state, action, reward, next_state, done)
            else:
                agent.update(state, action, reward, next_state, done, next_action)

            if reward == -100:
                falls += 1

            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        cliff_falls.append(falls)

    return agent, np.array(rewards), np.array(cliff_falls)


def trace_greedy_policy(agent, max_steps=80):
    env = agent.env
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset()
    path = [state]
    total_reward = 0
    falls = 0
    reached_goal = False

    for _ in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        if reward == -100:
            falls += 1

        path.append(next_state)
        total_reward += reward
        state = next_state

        if done:
            reached_goal = state == env.goal_state
            break

    agent.epsilon = old_epsilon
    return path, total_reward, falls, reached_goal


def count_cliff_edge_steps(path, env):
    edge_y = env.height - 2
    cliff_xs = range(1, env.width - 1)
    return sum(1 for y, x in path if y == edge_y and x in cliff_xs)


def render_policy(agent):
    env = agent.env
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}

    rows = []
    for y in range(env.height):
        cells = []
        for x in range(env.width):
            state = (y, x)
            if state == env.start_state:
                cells.append("S")
            elif state == env.goal_state:
                cells.append("G")
            elif state in env.cliff:
                cells.append("X")
            else:
                cells.append(arrows[agent.choose_action(state)])
        rows.append(" ".join(cells))

    agent.epsilon = old_epsilon
    return "\n".join(rows)


def print_report(name, agent, rewards, cliff_falls):
    path, path_reward, greedy_falls, reached_goal = trace_greedy_policy(agent)
    edge_steps = count_cliff_edge_steps(path, agent.env)
    recent = slice(-100, None)

    print(f"\n=== {name} ===")
    print(f"Average reward, last 100 episodes: {np.mean(rewards[recent]):.1f}")
    print(f"Average cliff falls, last 100 episodes: {np.mean(cliff_falls[recent]):.2f}")
    print(f"Greedy path length: {len(path) - 1}")
    print(f"Greedy path reward: {path_reward:.1f}")
    print(f"Greedy path reached goal: {reached_goal}")
    print(f"Greedy path cliff falls: {greedy_falls}")
    print(f"Greedy path edge-adjacent steps: {edge_steps}")
    print("\nGreedy policy:")
    print(render_policy(agent))


if __name__ == "__main__":
    q_agent, q_rewards, q_falls = train("q_learning", seed=7)
    sarsa_agent, sarsa_rewards, sarsa_falls = train("sarsa", seed=7)

    print("Cliff Walking diagnostic report")
    print("Expected pattern: Q-Learning is usually closer to the cliff edge; SARSA is usually more conservative.")

    print_report("Q-Learning", q_agent, q_rewards, q_falls)
    print_report("SARSA", sarsa_agent, sarsa_rewards, sarsa_falls)

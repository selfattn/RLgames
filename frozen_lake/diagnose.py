import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_lake.agent import PolicyIterationAgent, ValueIterationAgent
from frozen_lake.env import FrozenLakeEnv


def run_value_iteration(env, gamma=0.99, theta=1e-5, max_iterations=1000):
    agent = ValueIterationAgent(env, gamma=gamma, theta=theta)
    deltas = []

    for iteration in range(1, max_iterations + 1):
        delta = agent.step_value_iteration()
        deltas.append(delta)
        if delta < theta:
            break

    agent.extract_policy()
    return agent, deltas


def render_values(env, values):
    lines = []
    for y in range(env.height):
        cells = []
        for x in range(env.width):
            state_type = env.get_state_type(y, x)
            if state_type in ["H", "G"]:
                cells.append(f"{state_type:>6}")
            else:
                cells.append(f"{values[y, x]:6.3f}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def render_policy(env, policy):
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    lines = []

    for y in range(env.height):
        cells = []
        for x in range(env.width):
            state_type = env.get_state_type(y, x)
            if state_type == "H":
                cells.append("H")
            elif state_type == "G":
                cells.append("G")
            elif state_type == "S":
                cells.append(f"S{arrows[policy[y, x]]}")
            else:
                cells.append(arrows[policy[y, x]])
        lines.append(" ".join(f"{cell:>2}" for cell in cells))

    return "\n".join(lines)


def count_policy_mismatches(env, policy_a, policy_b):
    mismatches = 0
    compared = 0

    for y in range(env.height):
        for x in range(env.width):
            if env.get_state_type(y, x) in ["H", "G"]:
                continue

            compared += 1
            if policy_a[y, x] != policy_b[y, x]:
                mismatches += 1

    return mismatches, compared


if __name__ == "__main__":
    env = FrozenLakeEnv(slip_prob=0.1)

    value_agent, value_deltas = run_value_iteration(env)

    policy_agent = PolicyIterationAgent(env, gamma=0.99, theta=1e-5)
    policy_history = policy_agent.run_policy_iteration()

    value_diff = np.max(np.abs(value_agent.V - policy_agent.V))
    mismatches, compared = count_policy_mismatches(env, value_agent.policy, policy_agent.policy)

    print("Frozen Lake DP diagnostic report")
    print("Both methods use the exact transition model from env.get_transitions(state, action).")
    print()
    print("Value Iteration")
    print(f"- Bellman optimality sweeps: {len(value_deltas)}")
    print(f"- Final delta: {value_deltas[-1]:.8f}")
    print()
    print("Policy Iteration")
    print(f"- Policy improvement rounds: {len(policy_history)}")
    print(f"- Total policy-evaluation sweeps: {sum(item['eval_iterations'] for item in policy_history)}")
    print(f"- Last changed states: {policy_history[-1]['changed_states']}")
    print()
    print("Comparison")
    print(f"- Max |V_value_iteration - V_policy_iteration|: {value_diff:.8f}")
    print(f"- Greedy policy mismatches: {mismatches}/{compared}")
    print("  Note: value-equivalent ties can produce different arrows without changing optimality.")
    print()
    print("Value Iteration policy:")
    print(render_policy(env, value_agent.policy))
    print()
    print("Policy Iteration policy:")
    print(render_policy(env, policy_agent.policy))
    print()
    print("Final value function:")
    print(render_values(env, value_agent.V))

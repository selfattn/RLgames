import time
import os
import sys
import numpy as np
import argparse

# Make imports work from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frozen_lake.env import FrozenLakeEnv
from frozen_lake.agent import PolicyIterationAgent, ValueIterationAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_value_function(agent, title, iteration, delta=None, extra_lines=None):
    env = agent.env
    
    print(f"--- {title} ---")
    print(f"Iteration: {iteration}")
    if delta is not None:
        print(f"Max Change (Delta): {delta:.6f}")
    if extra_lines:
        for line in extra_lines:
            print(line)
    print()
    print("Value Matrix V(s):")
    
    # Render Value Function grid
    for y in range(env.height):
        row_str = ""
        for x in range(env.width):
            state_type = env.get_state_type(y, x)
            val = agent.V[y, x]
            
            if state_type == 'S':
                color = "\033[93m" # Yellow Start
            elif state_type == 'G':
                color = "\033[92m" # Green Goal
            elif state_type == 'H':
                color = "\033[91m" # Red Hole
            else:
                # Color intensity based on value
                color = f"\033[38;2;{int(val*255)};255;255m" if val > 0 else "\033[90m" # Grey for 0
                
            reset = "\033[0m"
            row_str += f"{color}{val:6.3f} [{state_type}]{reset}  "
        print(row_str)
        
    print("\nExtracted Optimal Policy:")
    # Render Policy grid
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    for y in range(env.height):
        row_str = ""
        for x in range(env.width):
            state_type = env.get_state_type(y, x)
            if state_type in ['H', 'G']:
                row_str += "   " # No policy for terminal states
            else:
                action = agent.policy[y, x]
                row_str += f" {arrow_map[action]} "
        print(row_str)
        
    print("\nWatch how the values propagate backwards from the Goal (G).")
    print("Values decrease as they get further from the goal due to discount factor gamma and slip probability.")

def run_value_iteration(delay):
    env = FrozenLakeEnv(slip_prob=0.1)
    agent = ValueIterationAgent(env, gamma=0.99)
    
    iteration = 0
    delta = float('inf')
    
    while delta > agent.theta:
        clear_screen()
        delta = agent.step_value_iteration()
        iteration += 1
        
        render_value_function(agent, "VALUE ITERATION (Dynamic Programming)", iteration, delta)
        time.sleep(delay)
        
    clear_screen()
    agent.extract_policy()
    render_value_function(agent, "VALUE ITERATION (Dynamic Programming)", iteration, delta)
    print("\nConvergence reached. The value function has stabilized.")
    print("The optimal policy tells the agent which direction to move at each state.")
    return agent, iteration, delta

def run_policy_iteration(delay):
    env = FrozenLakeEnv(slip_prob=0.1)
    agent = PolicyIterationAgent(env, gamma=0.99)
    
    policy_iteration = 0
    stable = False
    
    while not stable:
        policy_iteration += 1
        eval_iterations, delta = agent.evaluate_policy()
        stable, changed = agent.improve_policy()
        
        clear_screen()
        render_value_function(
            agent,
            "POLICY ITERATION (Dynamic Programming)",
            policy_iteration,
            delta,
            extra_lines=[
                f"Policy evaluation sweeps: {eval_iterations}",
                f"Policy improvement changed states: {changed}",
                f"Policy stable: {stable}",
            ],
        )
        time.sleep(delay)
        
    print("\nPolicy is stable. Policy Iteration has converged.")
    return agent, policy_iteration

def run_compare():
    env = FrozenLakeEnv(slip_prob=0.1)
    
    value_agent = ValueIterationAgent(env, gamma=0.99)
    value_iterations = 0
    delta = float('inf')
    while delta > value_agent.theta:
        delta = value_agent.step_value_iteration()
        value_iterations += 1
    value_agent.extract_policy()
    
    policy_agent = PolicyIterationAgent(env, gamma=0.99)
    policy_history = policy_agent.run_policy_iteration()
    
    value_diff = np.max(np.abs(value_agent.V - policy_agent.V))
    policy_mismatches = 0
    compared = 0
    for y in range(env.height):
        for x in range(env.width):
            if env.get_state_type(y, x) in ['H', 'G']:
                continue
            compared += 1
            if value_agent.policy[y, x] != policy_agent.policy[y, x]:
                policy_mismatches += 1
    
    render_value_function(value_agent, "VALUE ITERATION FINAL POLICY", value_iterations, delta)
    print("\n--- DP METHOD COMPARISON ---")
    print(f"Value Iteration sweeps: {value_iterations}")
    print(f"Policy Iteration improvement rounds: {len(policy_history)}")
    print(f"Policy Iteration evaluation sweeps: {sum(item['eval_iterations'] for item in policy_history)}")
    print(f"Max value difference: {value_diff:.8f}")
    print(f"Greedy policy mismatches: {policy_mismatches}/{compared}")

def parse_args():
    parser = argparse.ArgumentParser(description="Frozen Lake DP visualizer")
    parser.add_argument(
        "--mode",
        choices=["value", "policy", "compare"],
        default="value",
        help="Which dynamic programming demo to run.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds to pause between animation frames.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Welcome to RL Playground: Frozen Lake!")
    print("We will use Dynamic Programming to solve the 4x4 Grid.")
    print("This requires knowing the exact transition probabilities (Model-Based RL).\n")
    time.sleep(2)

    if args.mode == "value":
        run_value_iteration(args.delay)
    elif args.mode == "policy":
        run_policy_iteration(args.delay)
    else:
        run_compare()

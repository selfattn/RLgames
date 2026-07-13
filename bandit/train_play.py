import time
import os
import sys
import numpy as np

# Make imports work from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit.env import MultiArmedBanditEnv
from bandit.agent import EpsilonGreedyAgent, UCBAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def format_bar(val, max_val, length=20):
    """Utility to draw ASCII progress bars based on value"""
    if max_val == 0: return ""
    ratio = max(0, min(1.0, val / max_val))
    filled = int(ratio * length)
    return "█" * filled + "░" * (length - filled)

def run_experiment(agent_name, agent, env, steps=1000, delay=0.1):
    clear_screen()
    print(f"--- 🎰 MULTI-ARMED BANDIT: {agent_name.upper()} ---")
    print("Welcome to the Casino! The Agent must find the machine that pays out the most.")
    print("Every pull is random, making the true value hard to guess (Exploration vs. Exploitation).")
    print(f"Goal: Find Arm {env.best_action} (the true best arm).\n")
    time.sleep(3)
    
    # Store history for visualization
    history_rewards = []
    optimal_action_counts = 0
    
    # Define frames for animation
    # Only animate the first N steps, then jump to the end
    animate_steps = [10, 50, 100, 200, 500, 1000]
    
    for step in range(1, steps + 1):
        # 1. Agent chooses an arm based on its current knowledge
        action = agent.act()
        
        # 2. Environment provides a noisy reward
        reward = env.step(action)
        history_rewards.append(reward)
        
        if action == env.best_action:
            optimal_action_counts += 1
            
        # 3. Agent updates its internal estimates
        agent.learn(action, reward)
        
        # 4. Visualization Frame Update
        if step in animate_steps or step == steps:
            clear_screen()
            print(f"--- 🎰 {agent_name.upper()} | Step {step}/{steps} ---")
            print(f"Total Reward Collected: {sum(history_rewards):.2f}")
            print(f"% Optimal Action Chosen: {(optimal_action_counts / step) * 100:.1f}%\n")
            
            # Print the state of each arm
            print("Arm | True Mean | Agent's Q-Estimate | Times Pulled N(a)")
            print("-" * 65)
            
            true_means = env.get_true_means()
            # Normalize true means for drawing the bar to avoid negative bars messing up logic
            max_mean = np.max(np.abs(true_means)) + 0.1 
            
            for a in range(env.k_arms):
                true_val = true_means[a]
                est_val = agent.Q[a]
                pull_count = agent.N[a]
                
                # Highlight the true best arm in Green
                if a == env.best_action:
                    color = "\033[92m" # Green
                else:
                    color = "\033[0m" # Default
                    
                reset = "\033[0m"
                
                # Highlight the arm the agent thinks is best in Blue
                agent_best = np.argmax(agent.Q)
                est_color = "\033[94m" if a == agent_best else "\033[0m"
                
                print(f"{color} {a}  |  {true_val:7.2f}  {reset}| {est_color}{est_val:16.2f}{reset} | {pull_count:6d} {format_bar(pull_count, step, 15)}")
                
            print("\nGreen row = The *actual* best machine.")
            print("Blue text = The machine the *agent thinks* is best right now.")
            
            time.sleep(delay * (2 if step < 100 else 1.5)) # Slow down early frames
            
    return sum(history_rewards), optimal_action_counts / steps

if __name__ == "__main__":
    np.random.seed(42) # Set seed for reproducibility across the two agents
    k_arms = 5
    
    # Create the SAME environment (casino) for both agents to ensure a fair comparison
    # We save the true values and initialize two identical envs
    master_env = MultiArmedBanditEnv(k_arms=k_arms, mean=0.0, std=1.0)
    true_values = master_env.get_true_means()
    
    env_epsilon = MultiArmedBanditEnv(k_arms=k_arms)
    env_epsilon.true_action_values = np.copy(true_values)
    env_epsilon.best_action = np.argmax(true_values)
    
    env_ucb = MultiArmedBanditEnv(k_arms=k_arms)
    env_ucb.true_action_values = np.copy(true_values)
    env_ucb.best_action = np.argmax(true_values)
    
    # 1. Run Epsilon-Greedy Agent
    agent_eps = EpsilonGreedyAgent(k_arms=k_arms, epsilon=0.1)
    eps_reward, eps_opt_pct = run_experiment("Epsilon-Greedy (eps=0.1)", agent_eps, env_epsilon, steps=1000, delay=1.0)
    
    time.sleep(2)
    
    # 2. Run UCB Agent
    agent_ucb = UCBAgent(k_arms=k_arms, c=2.0)
    ucb_reward, ucb_opt_pct = run_experiment("UCB (c=2.0)", agent_ucb, env_ucb, steps=1000, delay=1.0)
    
    time.sleep(1)
    clear_screen()
    print("--- 🏆 FINAL SHOWDOWN 🏆 ---")
    print("\nEpsilon-Greedy (eps=0.1):")
    print(f"Total Reward: {eps_reward:.2f} | Optimal Action %: {eps_opt_pct*100:.1f}%")
    
    print("\nUpper Confidence Bound (c=2.0):")
    print(f"Total Reward: {ucb_reward:.2f} | Optimal Action %: {ucb_opt_pct*100:.1f}%")
    
    print("\nDid you notice?")
    print("1. Epsilon-Greedy wastes 10% of its time exploring randomly, even after it has found the best arm.")
    print("2. UCB explores deterministically. It tries arms it hasn't pulled in a while, but focuses heavily on the best arm once confident.")
    print("   (UCB usually achieves a higher optimal action % in the long run).")

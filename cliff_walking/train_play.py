import time
import os
import sys
import numpy as np

# Make imports work from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cliff_walking.env import CliffWalkingEnv
from cliff_walking.agent import QLearningAgent, SARSAAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def train_agent(agent_type, episodes=500, max_steps=500):
    env = CliffWalkingEnv()
    if agent_type == 'q_learning':
        agent = QLearningAgent(env, alpha=0.5, epsilon=0.1)
    else:
        agent = SARSAAgent(env, alpha=0.5, epsilon=0.1)
        
    print(f"Training {agent_type.upper()} for {episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state) # For SARSA, we need the initial action
        total_reward = 0
        done = False
        
        for _ in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state) # Determine next action early
            
            # The core difference in updating
            if agent_type == 'q_learning':
                # Q-Learning doesn't care what next_action is actually chosen, it looks for max Q
                agent.update(state, action, reward, next_state, done)
            else:
                # SARSA explicitly passes the next_action that was chosen (including potential epsilon exploration)
                agent.update(state, action, reward, next_state, done, next_action)
                
            state = next_state
            action = next_action
            total_reward += reward

            if done:
                break
            
        total_rewards.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
            
    print("Training finished.")
    return agent, total_rewards

def play_animation(agent, title, delay=0.2):
    env = agent.env
    state = env.reset()
    # Turn off exploration for the demonstration walk
    agent.epsilon = 0.0
    action = agent.choose_action(state)
    done = False
    step_count = 0
    fell_count = 0
    
    while not done and step_count < 30: # Limit steps to avoid infinite loops if it gets stuck
        clear_screen()
        print(f"--- {title} ---")
        print(f"Watch how the {title.split()[0]} algorithm navigates the cliff!")
        print("-" * 30)
        print(env.render())
        print(f"Step: {step_count}")
        
        next_state, reward, done = env.step(action)
        if reward == -100:
            fell_count += 1
        next_action = agent.choose_action(next_state)
        
        state = next_state
        action = next_action
        step_count += 1
        time.sleep(delay)
        
    clear_screen()
    print(f"--- {title} ---")
    print(env.render())
    if state == env.goal_state:
        print(f"🎉 Success! Reached the goal in {step_count} steps.")
    elif fell_count > 0:
        print(f"Fell off the cliff {fell_count} time(s) and did not reach the goal within the step limit.")
    else:
        print("Did not reach the goal within the step limit.")
    time.sleep(2)

if __name__ == "__main__":
    print("Welcome to RL Playground: Cliff Walking!")
    print("We will train two agents: Q-Learning and SARSA.")
    print("Watch the terminal animation to see the difference in their learned policies.\n")
    time.sleep(2)
    
    # Train agents
    q_agent, _ = train_agent('q_learning', episodes=500)
    sarsa_agent, _ = train_agent('sarsa', episodes=500)
    
    print("\nTraining complete! Preparing the live demonstration...")
    time.sleep(2)
    
    # Demonstrate policies
    play_animation(q_agent, "Q-LEARNING (Off-policy): The Risk Taker", delay=0.3)
    
    print("\nNow let's see SARSA...\n")
    time.sleep(2)
    
    play_animation(sarsa_agent, "SARSA (On-policy): The Cautious Explorer", delay=0.3)
    
    print("\n--- Conclusion ---")
    print("Did you notice?")
    print("1. Q-Learning learned the shortest path right along the edge of the cliff.")
    print("   (Because it assumes it will always take the optimal action in the future).")
    print("2. SARSA learned a longer, safer path far away from the cliff.")
    print("   (Because it knows it occasionally 'explores' randomly, and exploring near a cliff is fatal).")

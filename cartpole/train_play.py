import os
import sys
import time
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cartpole.env import CartPoleEnv
from cartpole.agent import DQNAgent

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(num_episodes=250, render_every=50, max_steps=500):
    env = CartPoleEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    episode_rewards = []
    
    print("🚀 Starting DQN Training on CartPole...")
    print("Phase 1: Agent starts with epsilon=1.0 (pure random).")
    print("Phase 2: As it gathers experience and learns, epsilon decays.")
    print("Phase 3: Eventually, the pole should balance for many steps.\n")
    time.sleep(2)
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        should_render = (episode % render_every == 0) or episode == 1 or episode == num_episodes
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            # Periodically render training to show progress
            if should_render and step % 2 == 0:
                clear_screen()
                print(f"🎮 Episode {episode}/{num_episodes} | Step {step} | Reward: {total_reward:.0f}")
                print(f"Epsilon (exploration rate): {agent.epsilon:.3f}")
                print(f"Replay Buffer: {len(agent.replay_buffer)} / {agent.replay_buffer.buffer.maxlen}")
                print()
                print(env.render(width=50))
                print()
                recent = episode_rewards[-10:] if episode_rewards else [0]
                print(f"Last 10 episodes' avg reward: {np.mean(recent):.1f}")
                time.sleep(0.02)
                
            if done:
                break
                
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        # Non-rendering episode: just print the progress line
        if not should_render and episode % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            bar = "█" * int(min(avg, max_steps) / max_steps * 30)
            print(f"Episode {episode:4d} | Avg Reward (last 10): {avg:6.1f} | Epsilon: {agent.epsilon:.3f} {bar}")
            
    return agent, episode_rewards

def demonstrate(agent, max_steps=500, delay=0.05):
    """Show the trained agent playing with zero exploration."""
    env = CartPoleEnv()
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        clear_screen()
        print(f"🏆 TRAINED AGENT DEMONSTRATION | Step {step} | Reward: {total_reward:.0f}")
        print("Epsilon=0.0 (pure exploitation)\n")
        print(env.render(width=50))
        
        action = agent.act(state, greedy=True)
        state, reward, done = env.step(action)
        total_reward += reward
        time.sleep(delay)
        
        if done:
            break
            
    clear_screen()
    print(f"🎯 Final Result: Agent balanced the pole for {int(total_reward)} steps!")
    print(env.render(width=50))

if __name__ == "__main__":
    set_seed(42)
    
    print("=" * 60)
    print("Welcome to RL Playground: CartPole + DQN")
    print("=" * 60)
    print("""
The Challenge: Balance a pole on a moving cart.
- State (4 continuous numbers): cart position, cart velocity, pole angle, angular velocity
- Action (2 discrete): push LEFT or push RIGHT
- Reward: +1 per step survived

Why a Neural Network?
Unlike Cliff Walking, the state space here is CONTINUOUS and INFINITE.
A Q-Table can't store infinite states, so we use a neural network to APPROXIMATE Q-values.

Two key tricks that make it work:
1. Replay Buffer: Store past (s, a, r, s') into a big pool, then train on random minibatches.
   -> Breaks correlation between consecutive samples.
2. Target Network: A slowly-updated copy of the Q-network used to compute targets.
   -> Prevents the 'moving target' problem.
""")
    time.sleep(4)
    
    agent, rewards = train(num_episodes=250, render_every=50, max_steps=500)
    
    print("\n" + "=" * 60)
    print("Training complete! Showing the trained agent in action...")
    print("=" * 60)
    time.sleep(2)
    
    demonstrate(agent, max_steps=500, delay=0.05)
    
    print("\n--- Takeaways ---")
    print("1. In the early episodes, the pole falls immediately (random actions).")
    print("2. After ~100 episodes, the DQN has learned a decent balancing strategy.")
    print("3. At the end, the trained agent can balance the pole for 500+ steps.")
    print(f"Max reward achieved: {max(rewards):.0f}")
    print(f"Average reward of last 20 episodes: {np.mean(rewards[-20:]):.1f}")

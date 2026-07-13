import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        # Q-table dimensions: (height, width, num_actions)
        self.Q = np.zeros((env.height, env.width, len(env.actions)))
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.actions = env.actions
        
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions) # Explore (random action)
        else:
            y, x = state
            q_values = self.Q[y, x, :]
            # Tie-breaking if multiple max actions
            max_indices = np.where(q_values == np.max(q_values))[0]
            return np.random.choice(max_indices) # Exploit (greedy action)
            
    def update(self, state, action, reward, next_state, done):
        y, x = state
        ny, nx = next_state
        
        # Q-Learning (Off-policy): TD Target uses the *max* Q-value of the next state
        # The agent *learns* about the optimal policy, even if it's currently *acting* epsilon-greedily
        best_next_action = np.argmax(self.Q[ny, nx, :])
        td_target = reward + self.gamma * self.Q[ny, nx, best_next_action] * (not done)
        
        td_error = td_target - self.Q[y, x, action]
        self.Q[y, x, action] += self.alpha * td_error

class SARSAAgent:
    def __init__(self, env, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        self.Q = np.zeros((env.height, env.width, len(env.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = env.actions
        
    def choose_action(self, state):
        # Same Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            y, x = state
            q_values = self.Q[y, x, :]
            max_indices = np.where(q_values == np.max(q_values))[0]
            return np.random.choice(max_indices)
            
    def update(self, state, action, reward, next_state, done, next_action=None):
        y, x = state
        ny, nx = next_state
        
        # SARSA (On-policy): TD Target uses the Q-value of the *actual* next action chosen by epsilon-greedy
        # The agent learns about the policy it is *currently following* (which includes exploration mistakes)
        td_target = reward + self.gamma * self.Q[ny, nx, next_action] * (not done)
        
        td_error = td_target - self.Q[y, x, action]
        self.Q[y, x, action] += self.alpha * td_error

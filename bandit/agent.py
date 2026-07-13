import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, k_arms=5, epsilon=0.1):
        self.k_arms = k_arms
        self.epsilon = epsilon
        # Q(a) estimates the value of action a
        self.Q = np.zeros(k_arms)
        # N(a) counts how many times action a was chosen
        self.N = np.zeros(k_arms, dtype=int)
        
    def act(self):
        # Epsilon probability to explore randomly
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.k_arms)
        else:
            # 1-Epsilon probability to exploit the best known action
            # Break ties randomly instead of always choosing the first max
            max_value = np.max(self.Q)
            best_actions = np.where(self.Q == max_value)[0]
            action = np.random.choice(best_actions)
            
        return action
        
    def learn(self, action, reward):
        # Update count N(a)
        self.N[action] += 1
        
        # Update estimate Q(a) using the incremental average formula:
        # NewEstimate = OldEstimate + StepSize * (Target - OldEstimate)
        # Here, StepSize is 1/N(a)
        step_size = 1.0 / self.N[action]
        error = reward - self.Q[action]
        self.Q[action] += step_size * error

class UCBAgent:
    """
    Upper Confidence Bound (UCB) Agent
    Instead of epsilon-randomness, UCB explicitly measures uncertainty.
    It prefers actions that haven't been tried much.
    """
    def __init__(self, k_arms=5, c=2.0):
        self.k_arms = k_arms
        self.c = c # Degree of exploration (higher = more exploration)
        self.Q = np.zeros(k_arms)
        self.N = np.zeros(k_arms, dtype=int)
        self.t = 0 # Total steps taken
        
    def act(self):
        self.t += 1
        
        # If there are any actions we haven't tried yet, try them first
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            action = np.random.choice(untried)
            return action
            
        # Calculate UCB score for all actions: Q(a) + c * sqrt(ln(t) / N(a))
        # The second term is the "uncertainty bonus"
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.N)
        ucb_scores = self.Q + uncertainty
        
        # Break ties randomly
        max_score = np.max(ucb_scores)
        best_actions = np.where(ucb_scores == max_score)[0]
        action = np.random.choice(best_actions)
            
        return action
        
    def learn(self, action, reward):
        self.N[action] += 1
        step_size = 1.0 / self.N[action]
        error = reward - self.Q[action]
        self.Q[action] += step_size * error

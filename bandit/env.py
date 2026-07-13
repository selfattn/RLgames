import numpy as np

class MultiArmedBanditEnv:
    def __init__(self, k_arms=5, mean=0.0, std=1.0):
        self.k_arms = k_arms
        # The true mean reward (q*) for each arm is chosen from a normal distribution
        # In a real casino, you wouldn't know these numbers!
        self.true_action_values = np.random.normal(mean, std, k_arms)
        self.best_action = np.argmax(self.true_action_values)
        
    def step(self, action):
        """
        Pulling an arm (action) returns a reward.
        The reward is sampled from a normal distribution around that arm's true mean.
        This represents the noise/variance of the real world.
        """
        # Reward = Normal(true_mean[action], variance=1)
        reward = np.random.normal(self.true_action_values[action], 1.0)
        return reward
        
    def get_true_means(self):
        return self.true_action_values

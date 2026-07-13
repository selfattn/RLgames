import numpy as np


def expected_action_value(env, V, state, action, gamma):
    q_val = 0.0
    transitions = env.get_transitions(state, action)
    for prob, next_state, reward, done in transitions:
        ny, nx = next_state
        q_val += prob * (reward + gamma * V[ny, nx])
    return q_val


def greedy_action_values(env, V, state, gamma):
    return np.array([
        expected_action_value(env, V, state, action, gamma)
        for action in env.actions
    ])


class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-5):
        self.env = env
        self.gamma = gamma # Discount factor
        self.theta = theta # Convergence threshold
        
        # Initialize Value function V(s) to zeros
        self.V = np.zeros((env.height, env.width))
        self.policy = np.zeros((env.height, env.width), dtype=int)
        
    def step_value_iteration(self):
        """
        Performs a single sweep of Value Iteration over all states.
        Returns the maximum change in value (delta).
        """
        delta = 0
        new_V = np.copy(self.V)
        
        for y in range(self.env.height):
            for x in range(self.env.width):
                state = (y, x)
                # Terminal states have value 0
                if self.env.get_state_type(y, x) in ['H', 'G']:
                    new_V[y, x] = 0
                    continue
                    
                # Bellman Optimality Equation:
                # V(s) = max_a sum[ P(s'|s,a) * (R + gamma * V(s')) ]
                action_values = greedy_action_values(self.env, self.V, state, self.gamma)
                    
                # Update value function with the maximum action value
                best_value = np.max(action_values)
                new_V[y, x] = best_value
                
                # Extract the greedy policy
                self.policy[y, x] = np.argmax(action_values)
                
                # Track convergence
                delta = max(delta, abs(best_value - self.V[y, x]))
                
        self.V = new_V
        return delta

    def extract_policy(self):
        for y in range(self.env.height):
            for x in range(self.env.width):
                state = (y, x)
                if self.env.get_state_type(y, x) in ['H', 'G']:
                    continue

                action_values = greedy_action_values(self.env, self.V, state, self.gamma)
                self.policy[y, x] = np.argmax(action_values)

        return self.policy


class PolicyIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-5):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.V = np.zeros((env.height, env.width))
        self.policy = np.zeros((env.height, env.width), dtype=int)

    def evaluate_policy(self, max_iterations=1000):
        """
        Iteratively evaluates the current policy until V^pi converges.
        Returns (iterations, final_delta).
        """
        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            new_V = np.copy(self.V)

            for y in range(self.env.height):
                for x in range(self.env.width):
                    state = (y, x)
                    if self.env.get_state_type(y, x) in ['H', 'G']:
                        new_V[y, x] = 0.0
                        continue

                    action = self.policy[y, x]
                    value = expected_action_value(self.env, self.V, state, action, self.gamma)
                    new_V[y, x] = value
                    delta = max(delta, abs(value - self.V[y, x]))

            self.V = new_V
            if delta < self.theta:
                return iteration, delta

        return max_iterations, delta

    def improve_policy(self):
        """
        Greedifies the policy with respect to the current value function.
        Returns (policy_stable, changed_state_count).
        """
        policy_stable = True
        changed = 0

        for y in range(self.env.height):
            for x in range(self.env.width):
                state = (y, x)
                if self.env.get_state_type(y, x) in ['H', 'G']:
                    continue

                old_action = self.policy[y, x]
                action_values = greedy_action_values(self.env, self.V, state, self.gamma)
                best_action = np.argmax(action_values)
                self.policy[y, x] = best_action

                if old_action != best_action:
                    policy_stable = False
                    changed += 1

        return policy_stable, changed

    def run_policy_iteration(self, max_policy_iterations=100):
        history = []

        for outer_iteration in range(1, max_policy_iterations + 1):
            eval_iterations, delta = self.evaluate_policy()
            policy_stable, changed = self.improve_policy()
            history.append({
                "policy_iteration": outer_iteration,
                "eval_iterations": eval_iterations,
                "delta": delta,
                "changed_states": changed,
                "policy_stable": policy_stable,
            })

            if policy_stable:
                break

        return history

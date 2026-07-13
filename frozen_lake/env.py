import numpy as np

class FrozenLakeEnv:
    def __init__(self, slip_prob=0.2):
        # 4x4 Grid
        # S: Start, F: Frozen (safe), H: Hole (trap), G: Goal
        self.desc = [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ]
        self.height = len(self.desc)
        self.width = len(self.desc[0])
        
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.actions = [0, 1, 2, 3]
        self.slip_prob = slip_prob  # Probability of slipping to an orthogonal direction
        
    def get_state_type(self, y, x):
        return self.desc[y][x]
        
    def get_transitions(self, state, action):
        """
        Returns a list of (probability, next_state, reward, done) for a given state and action.
        This represents the Environment Model P(s', r | s, a) used in Dynamic Programming.
        """
        y, x = state
        transitions = []
        
        # If already at a terminal state, no further transitions
        if self.get_state_type(y, x) in ['H', 'G']:
            return [(1.0, state, 0.0, True)]
            
        # Intended direction + 2 orthogonal slip directions
        intended = action
        slip1 = (action - 1) % 4
        slip2 = (action + 1) % 4
        
        for dir_action, prob in [(intended, 1.0 - self.slip_prob), 
                                 (slip1, self.slip_prob / 2), 
                                 (slip2, self.slip_prob / 2)]:
            
            ny, nx = y, x
            if dir_action == 0:   ny = max(y - 1, 0)
            elif dir_action == 1: nx = min(x + 1, self.width - 1)
            elif dir_action == 2: ny = min(y + 1, self.height - 1)
            elif dir_action == 3: nx = max(x - 1, 0)
            
            next_state = (ny, nx)
            state_type = self.get_state_type(ny, nx)
            
            done = state_type in ['H', 'G']
            reward = 1.0 if state_type == 'G' else 0.0
            
            transitions.append((prob, next_state, reward, done))
            
        return transitions

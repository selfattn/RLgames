import numpy as np

class CliffWalkingEnv:
    def __init__(self, width=12, height=4):
        self.width = width
        self.height = height
        # Actions: 0: up, 1: right, 2: down, 3: left
        self.actions = [0, 1, 2, 3]
        
        # State represents (y, x) coordinate
        self.start_state = (self.height - 1, 0)
        self.goal_state = (self.height - 1, self.width - 1)
        
        # Cliff area is at the bottom, excluding start and goal
        self.cliff = [(self.height - 1, x) for x in range(1, self.width - 1)]
        
        self.current_state = self.start_state
        
    def reset(self):
        self.current_state = self.start_state
        return self.current_state
        
    def step(self, action):
        y, x = self.current_state
        
        # Calculate next position
        if action == 0:   # up
            y = max(y - 1, 0)
        elif action == 1: # right
            x = min(x + 1, self.width - 1)
        elif action == 2: # down
            y = min(y + 1, self.height - 1)
        elif action == 3: # left
            x = max(x - 1, 0)
            
        next_state = (y, x)
        
        # Check if fell off cliff. In the classic Cliff Walking task, falling
        # sends the agent back to start, but the episode continues.
        if next_state in self.cliff:
            reward = -100
            done = False
            next_state = self.start_state # Teleport back to start
        # Check if reached goal
        elif next_state == self.goal_state:
            reward = 0  # In classic formulation, stepping into goal is -1 or 0
            done = True
        else:
            reward = -1 # Cost of living / step cost
            done = False
            
        self.current_state = next_state
        return next_state, reward, done
        
    def render(self):
        grid = ""
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == self.current_state:
                    grid += "🤖 " # Agent
                elif (y, x) == self.goal_state:
                    grid += "🏆 " # Goal
                elif (y, x) in self.cliff:
                    grid += "🌊 " # Cliff
                elif (y, x) == self.start_state:
                    grid += "🏠 " # Start
                else:
                    grid += "⬜ " # Safe ground
            grid += "\n"
        return grid

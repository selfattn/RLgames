import math
import numpy as np

class CartPoleEnv:
    """
    A hand-written CartPole physics environment (inspired by OpenAI Gym's classic control).
    
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Action: 0 (push cart to the left) or 1 (push cart to the right)
    Reward: +1 for every step the pole stays upright
    Done: pole tilts more than 12 degrees OR cart moves out of the ±2.4 track
    """
    def __init__(self):
        # Physics constants
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # half-length of the pole
        self.pole_mass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step between state updates (seconds)
        
        # Failure thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12 degrees
        self.x_threshold = 2.4
        
        self.state = None
        self.state_dim = 4
        self.action_dim = 2
        
    def reset(self):
        # Small random initial state near equilibrium
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.copy()
        
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force based on action
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # Classical equations of motion (see Sutton & Barto, 1998)
        temp = (force + self.pole_mass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass
        
        # Euler integration to update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check termination
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        reward = 1.0  # Always +1 for surviving one more step
        return self.state.copy(), reward, done
        
    def render(self, width=40):
        """Render a simple ASCII view of the CartPole."""
        x, _, theta, _ = self.state
        
        # Map cart position to column index
        cart_col = int((x + self.x_threshold) / (2 * self.x_threshold) * width)
        cart_col = max(0, min(width - 1, cart_col))
        
        # Pole top position (projected)
        pole_top_col = cart_col + int(math.sin(theta) * 8)
        pole_top_row_from_cart = int(math.cos(theta) * 8)  # How many rows up the tip is
        pole_top_row_from_cart = max(1, pole_top_row_from_cart)
        
        # Build ASCII frame (10 rows tall)
        frame_height = 10
        frame = [[' '] * width for _ in range(frame_height)]
        
        # Draw pole line from cart (row frame_height-1) upwards
        cart_row = frame_height - 1
        for i in range(1, pole_top_row_from_cart + 1):
            row = cart_row - i
            # Linear interpolation from cart to pole tip
            ratio = i / max(pole_top_row_from_cart, 1)
            col = int(cart_col + (pole_top_col - cart_col) * ratio)
            col = max(0, min(width - 1, col))
            if 0 <= row < frame_height:
                frame[row][col] = '|' if abs(pole_top_col - cart_col) < 2 else '/' if pole_top_col > cart_col else '\\'
        
        # Draw cart
        for dc in [-1, 0, 1]:
            c = cart_col + dc
            if 0 <= c < width:
                frame[cart_row][c] = '='
                
        # Build output string with borders
        lines = ["+" + "-" * width + "+"]
        for row in frame:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * width + "+")
        lines.append(f" x={x:+.2f}  theta={math.degrees(theta):+.2f}°")
        return "\n".join(lines)

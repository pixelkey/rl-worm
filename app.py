import pygame
import random
import math
import numpy as np
from models.dqn import WormAgent
from analytics.metrics import WormAnalytics
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the intelligent worm simulation')
parser.add_argument('--demo', action='store_true', help='Run in demo mode using the best trained model')
args = parser.parse_args()

class WormGame:
    def __init__(self):
        pygame.init()
        
        # Get the display info for full screen
        display_info = pygame.display.Info()
        self.screen_width = display_info.current_w
        self.screen_height = display_info.current_h - 60  # Small margin for window decorations
        
        # Set up the full screen window
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Worm" + (" (Demo Mode)" if args.demo else ""))
        
        # Calculate the game area (square in the middle)
        max_game_size = min(self.screen_width, self.screen_height)
        self.game_size = int(max_game_size * 0.8)  # 80% of the available space
        self.game_x_offset = (self.screen_width - self.game_size) // 2
        self.game_y_offset = (self.screen_height - self.game_size) // 2
        
        # Create a surface for the game area
        self.game_surface = pygame.Surface((self.game_size, self.game_size))
        
        # Worm properties - scale based on game area
        self.segment_length = int(self.game_size/60)
        self.segment_width = int(self.game_size/80)
        self.num_segments = 20
        self.head_size = int(self.game_size/53)
        self.spacing = 0.7
        
        # Colors
        self.head_color = (150, 50, 50)    # Reddish head
        self.body_colors = []
        for i in range(self.num_segments - 1):
            green_val = int(150 - (100 * i / (self.num_segments - 1)))
            self.body_colors.append((70, green_val + 30, 20))
        
        # Game boundaries (now using game_size)
        self.width = self.game_size
        self.height = self.game_size
        
        # Initialize positions list
        self.positions = []
        self.reset()
        
    def reset(self):
        self.x = self.width // 2
        self.y = self.height // 2
        self.angle = 0
        self.speed = 5
        self.positions = [(self.x, self.y)] * self.num_segments
        return self._get_state()
    
    def _draw_segment(self, pos, angle, size, color, is_head=False):
        # Create a surface for the segment
        segment_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        
        if is_head:
            # Draw head with eyes
            pygame.draw.ellipse(segment_surface, color, (0, 0, size * 2, size * 2))
            # Left eye
            pygame.draw.circle(segment_surface, (255, 255, 255), (size * 0.7, size), size//4)
            pygame.draw.circle(segment_surface, (0, 0, 0), (size * 0.7, size), size//8)
            # Right eye
            pygame.draw.circle(segment_surface, (255, 255, 255), (size * 1.3, size), size//4)
            pygame.draw.circle(segment_surface, (0, 0, 0), (size * 1.3, size), size//8)
        else:
            # Draw body segment
            pygame.draw.ellipse(segment_surface, color, (0, 0, size * 2, size * 2))
            # Add segment detail
            highlight_color = tuple(min(c + 20, 255) for c in color)
            pygame.draw.ellipse(segment_surface, highlight_color, 
                              (size * 0.5, size * 0.5, size, size))
        
        # Rotate the segment
        rotated_segment = pygame.transform.rotate(segment_surface, -angle * 180 / math.pi)
        segment_rect = rotated_segment.get_rect(center=pos)
        
        # Draw the segment onto the game surface (not the main screen)
        self.game_surface.blit(rotated_segment, segment_rect)
    
    def draw(self):
        # Fill the main screen with black
        self.screen.fill((0, 0, 0))
        
        # Fill the game surface with light gray
        self.game_surface.fill((240, 240, 240))
        
        # Draw body segments first (in reverse so head appears on top)
        for i in range(len(self.positions) - 1, 0, -1):
            pos = self.positions[i]
            prev_pos = self.positions[i-1]
            # Calculate angle between segments
            angle = math.atan2(prev_pos[1] - pos[1], prev_pos[0] - pos[0])
            self._draw_segment(pos, angle, self.segment_width, self.body_colors[i-1])
        
        # Draw head
        self._draw_segment(self.positions[0], self.angle, self.head_size, self.head_color, True)
        
        # Draw the game surface onto the main screen with offset
        self.screen.blit(self.game_surface, (self.game_x_offset, self.game_y_offset))
        
        # Draw border around game area
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (self.game_x_offset-2, self.game_y_offset-2, 
                         self.game_size+4, self.game_size+4), 2)
        
        pygame.display.flip()
    
    def step(self, action):
        wall_collision = False  # Initialize wall_collision
        
        # Update head position
        if action < 8:  # Movement actions
            self.angle = action * math.pi / 4
            dx = math.cos(self.angle) * self.speed
            dy = math.sin(self.angle) * self.speed
            new_x = max(self.head_size, min(self.width - self.head_size, self.x + dx))
            new_y = max(self.head_size, min(self.height - self.head_size, self.y + dy))
            
            # Check if we hit a wall
            wall_collision = (new_x in (self.head_size, self.width - self.head_size) or 
                            new_y in (self.head_size, self.height - self.head_size))
            
            self.x, self.y = new_x, new_y
        
        # Update position history for body segments
        self.positions.insert(0, (self.x, self.y))
        if len(self.positions) > self.num_segments:
            self.positions.pop()
        
        # Ensure smooth segment following
        if len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                prev_pos = self.positions[i-1]
                curr_pos = self.positions[i]
                dx = prev_pos[0] - curr_pos[0]
                dy = prev_pos[1] - curr_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > self.segment_width * self.spacing:
                    ratio = (self.segment_width * self.spacing) / dist
                    new_x = prev_pos[0] - dx * ratio
                    new_y = prev_pos[1] - dy * ratio
                    self.positions[i] = (new_x, new_y)
        
        return self._get_state(), wall_collision

    def _get_state(self):
        # Normalize positions to [0,1] range for the neural network
        norm_x = self.x / self.width
        norm_y = self.y / self.height
        
        # Calculate normalized distances to walls
        dist_right = (self.width - self.x) / self.width
        dist_left = self.x / self.width
        dist_bottom = (self.height - self.y) / self.height
        dist_top = self.y / self.height
        
        # Add velocity components
        if len(self.positions) > 1:
            prev_x, prev_y = self.positions[1]
            vel_x = (self.x - prev_x) / self.speed
            vel_y = (self.y - prev_y) / self.speed
        else:
            vel_x = vel_y = 0
            
        return [norm_x, norm_y, dist_right, dist_left, dist_bottom, dist_top, vel_x, vel_y]

# ML Agent setup
STATE_SIZE = 8  # position (2), velocity (2), distances to walls (4)
ACTION_SIZE = 9  # 8 directions + no movement
agent = WormAgent(STATE_SIZE, ACTION_SIZE)

# Load the best model if in demo mode
if args.demo:
    agent.load_state(use_best=True)
    agent.epsilon = 0.01  # Very little exploration in demo mode
    print("Running in demo mode with best trained model")

analytics = WormAnalytics()

# Episode tracking
episode = 0
steps_in_episode = 0
MAX_STEPS = 2000
positions_history = []

# Movement tracking
wall_collisions = 0
total_distance = 0
last_position = (0, 0)

def update_metrics():
    global total_distance, last_position, episode
    
    # Calculate metrics
    current_position = (game.x, game.y)
    distance = math.sqrt((current_position[0] - last_position[0])**2 + 
                        (current_position[1] - last_position[1])**2)
    total_distance += distance
    
    # Calculate exploration ratio (unique positions / total positions)
    unique_positions = len(set((int(p[0]), int(p[1])) for p in positions_history))
    exploration_ratio = unique_positions / len(positions_history) if positions_history else 0
    
    # Calculate movement smoothness
    if len(positions_history) > 2:
        vectors = np.diff(np.array(positions_history[-3:]), axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        smoothness = 1.0 / (1.0 + np.std(angles))
    else:
        smoothness = 1.0
    
    metrics_data = {
        'avg_speed': math.sqrt(game.speed**2),
        'wall_collisions': wall_collisions,
        'exploration_ratio': exploration_ratio,
        'movement_smoothness': smoothness
    }
    
    analytics.update_metrics(episode, metrics_data)
    last_position = current_position

game = WormGame()

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Get current state
    state = game._get_state()
    
    # Get action from agent
    action = agent.act(state)
    
    # Update physics with action
    next_state, wall_collision = game.step(action)
    
    # Calculate reward
    reward = -1  # Default reward
    
    # Store experience in memory
    agent.memory.push(state, action, reward, next_state, False)
    
    # Train the agent
    loss = agent.train()
    
    # Update metrics
    update_metrics()
    
    # Increment steps
    steps_in_episode += 1
    
    # Check if episode should end
    if steps_in_episode >= MAX_STEPS:
        # Save the model state every 5 episodes
        if episode % 5 == 0:
            agent.save_state(episode)
            print(f"Saved model state at episode {episode}")
        
        # Generate analytics report every 10 episodes
        if episode % 10 == 0:
            report_path = analytics.generate_report(episode)
            heatmap_path = analytics.plot_heatmap(positions_history, (game.width, game.height), episode)
            print(f"Generated report: {report_path}")
            print(f"Generated heatmap: {heatmap_path}")
        
        # Reset episode
        episode += 1
        steps_in_episode = 0
        positions_history = []
        wall_collisions = 0
        total_distance = 0
        game.reset()
        
        # Update target network periodically
        if episode % 5 == 0:
            agent.update_target_model()
    
    # Draw game
    game.draw()
    
    # Cap the frame rate
    clock.tick(60)

pygame.quit()
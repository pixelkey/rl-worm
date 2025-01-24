import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import argparse
import time
from datetime import datetime, timedelta
import sys
from analytics.metrics import WormAnalytics
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the intelligent worm simulation')
parser.add_argument('--demo', action='store_true', help='Run in demo mode using the best trained model')
args = parser.parse_args()

class Plant:
    def __init__(self, x, y, game_size):
        self.x = x
        self.y = y
        self.size = int(game_size/80)  # Plant size scales with game
        self.max_lifetime = 500  # How long the plant lives
        self.lifetime = self.max_lifetime
        self.state = 'growing'  # growing, mature, wilting
        self.color = (0, 180, 0)  # Start with bright green
        
    def update(self):
        self.lifetime -= 1
        if self.lifetime <= 0:
            return False  # Plant is dead
            
        # Update plant state and color based on lifetime
        if self.lifetime > self.max_lifetime * 0.7:
            self.state = 'growing'
            growth_factor = (self.max_lifetime - self.lifetime) / (self.max_lifetime * 0.3)
            self.color = (0, int(140 + 40 * growth_factor), 0)
        elif self.lifetime > self.max_lifetime * 0.3:
            self.state = 'mature'
            self.color = (0, 180, 0)
        else:
            self.state = 'wilting'
            wilt_factor = self.lifetime / (self.max_lifetime * 0.3)
            self.color = (int(139 * (1-wilt_factor)), int(69 * (1-wilt_factor)), 19)
            
        return True
        
    def draw(self, surface):
        if self.state == 'growing':
            size_factor = (self.max_lifetime - self.lifetime) / (self.max_lifetime * 0.3)
            current_size = int(self.size * min(1.0, size_factor))
        elif self.state == 'wilting':
            wilt_factor = self.lifetime / (self.max_lifetime * 0.3)
            current_size = int(self.size * wilt_factor)
        else:
            current_size = self.size
            
        # Draw stem
        stem_height = current_size * 2
        pygame.draw.line(surface, (0, 100, 0), 
                        (self.x, self.y + current_size),
                        (self.x, self.y + current_size + stem_height), 2)
        
        # Draw plant head
        pygame.draw.circle(surface, self.color, (self.x, self.y + current_size), current_size)

class WormGame:
    def __init__(self, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        
        # Set fixed dimensions for game area
        self.screen_width = 800
        self.screen_height = 600
        self.headless = headless
        
        # Create game surface and screen only if not headless
        if not headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("AI Worm" + (" (Demo Mode)" if args.demo else ""))
        else:
            pygame.display.set_mode((1, 1))  # Minimal display for headless mode
            
        self.game_surface = pygame.Surface((self.screen_width, self.screen_height))
        
        # Game area offset
        self.game_x_offset = 0
        self.game_y_offset = 0
        
        # Worm properties
        self.segment_length = int(self.screen_height/60)
        self.segment_width = int(self.screen_height/80)
        self.num_segments = 20
        self.max_segments = 30
        self.min_segments = 10
        self.head_size = int(self.screen_height/53)
        self.spacing = 0.7
        
        # Hunger and growth mechanics
        self.max_hunger = 1000
        self.hunger = self.max_hunger
        self.hunger_rate = 0.5
        self.hunger_gain_from_plant = 300
        
        # Plant mechanics
        self.plants = []
        self.plant_spawn_rate = 0.02
        self.max_plants = 5
        
        # Colors
        self.head_color = (150, 50, 50)
        self.body_colors = []
        for i in range(self.max_segments - 1):  # Pre-generate colors for max possible length
            green_val = int(150 - (100 * i / (self.max_segments - 1)))
            self.body_colors.append((70, green_val + 30, 20))
        
        # Game boundaries
        self.width = self.screen_width
        self.height = self.screen_height
        
        # Initialize worm
        self.reset()

    def spawn_plant(self):
        if len(self.plants) < self.max_plants and random.random() < self.plant_spawn_rate:
            margin = self.screen_height // 10
            x = random.randint(margin, self.screen_width - margin)
            y = random.randint(margin, self.screen_height - margin)
            
            # Check if too close to other plants
            too_close = False
            for plant in self.plants:
                dist = math.sqrt((x - plant.x)**2 + (y - plant.y)**2)
                if dist < self.screen_height // 8:
                    too_close = True
                    break
                    
            if not too_close:
                self.plants.append(Plant(x, y, self.screen_height))
                
    def update_plants(self):
        # Update existing plants
        self.plants = [plant for plant in self.plants if plant.update()]
        
        # Try to spawn new plant
        self.spawn_plant()
        
    def check_plant_collision(self):
        """Check for collisions with plants and handle eating"""
        head_rect = pygame.Rect(self.x - self.head_size/2, self.y - self.head_size/2, 
                              self.head_size, self.head_size)
        
        for plant in self.plants[:]:  # Use slice copy to safely modify during iteration
            plant_rect = pygame.Rect(plant.x - plant.size/2, plant.y - plant.size/2,
                                   plant.size, plant.size)
            
            if head_rect.colliderect(plant_rect):
                self.plants.remove(plant)
                self.hunger = min(self.max_hunger, self.hunger + self.hunger_gain_from_plant)
                
                # Grow when eating
                if self.num_segments < self.max_segments:
                    self.num_segments += 1
                    # Add new segment at the end
                    last_pos = self.positions[-1]
                    self.positions.append(last_pos)
                    # Add new color for the segment
                    green_val = int(150 - (100 * (self.num_segments-1) / (self.max_segments - 1)))
                    self.body_colors.append((70, green_val + 30, 20))
                
                return True
        return False
        
    def update_hunger(self):
        """Update hunger and return whether worm is still alive"""
        self.hunger = max(0, self.hunger - self.hunger_rate)
        
        # Calculate shrinking probability based on hunger level
        hunger_ratio = self.hunger / self.max_hunger
        if hunger_ratio < 0.5:  # Start shrinking below 50% hunger
            # Shrinking probability increases as hunger decreases
            # At 50% hunger: 0.1% chance per frame
            # At 0% hunger: 5% chance per frame
            shrink_probability = 0.05 * (1 - (hunger_ratio * 2))
            
            if self.num_segments > self.min_segments and random.random() < shrink_probability:
                self.num_segments -= 1
                if len(self.positions) > self.num_segments:
                    self.positions.pop()  # Remove last segment
                    self.body_colors.pop()  # Remove its color
        
        # Die if no segments left or hunger is zero
        return self.hunger > 0 and self.num_segments > 0
        
    def step(self, action):
        """Execute one time step within the environment"""
        # Convert action to movement
        angle = action * (2 * math.pi / 8)  # Convert to radians (8 possible directions)
        speed = 5.0  # Fixed speed
        
        # Calculate movement
        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed
        
        # Previous position for movement calculations
        prev_x = self.x
        prev_y = self.y
        
        # Update position
        self.x += dx
        self.y += dy
        
        # Check wall collisions and constrain position
        wall_collision = False
        margin = self.head_size
        if self.x < margin:
            self.x = margin
            wall_collision = True
        elif self.x > self.width - margin:
            self.x = self.width - margin
            wall_collision = True
        if self.y < margin:
            self.y = margin
            wall_collision = True
        elif self.y > self.height - margin:
            self.y = self.height - margin
            wall_collision = True
            
        # Calculate movement distance
        move_dist = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
        
        # Only update body if there was significant movement
        if move_dist > 0.1:  # Small threshold to prevent micro-movements
            # Update body segments
            self.positions[0] = (self.x, self.y)
            for i in range(1, len(self.positions)):
                target_x = self.positions[i-1][0]
                target_y = self.positions[i-1][1]
                curr_x = self.positions[i][0]
                curr_y = self.positions[i][1]
                
                # Calculate direction to target
                dx = target_x - curr_x
                dy = target_y - curr_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Move segment towards target, maintaining spacing
                if dist > self.segment_width * self.spacing:
                    move_ratio = (dist - self.segment_width * self.spacing) / dist
                    new_x = curr_x + dx * move_ratio
                    new_y = curr_y + dy * move_ratio
                    self.positions[i] = (new_x, new_y)
        
        # Update plants and check collisions
        self.update_plants()
        ate_plant = self.check_plant_collision()
        
        # Update hunger
        alive = self.update_hunger()
        
        # Calculate reward
        reward = 0
        
        # Reward for eating plants and growing
        if ate_plant:
            # Higher reward when hungrier
            hunger_ratio = 1 - (self.hunger / self.max_hunger)
            reward += 10.0 * (1 + hunger_ratio)  # Base reward ranges from 10 to 20 based on hunger
            
            # Extra reward for growing
            if self.num_segments > len(self.positions) - 1:  # If we grew
                reward += 5.0  # Bonus for growing
                
        # Penalty for hitting walls
        if wall_collision:
            reward -= 1.0
        
        # Penalty for shrinking (more severe as we get shorter)
        if self.num_segments < len(self.positions) - 1:  # If we shrunk
            # Penalty increases as we get shorter
            length_ratio = self.num_segments / self.max_segments
            reward -= 2.0 * (1 + (1 - length_ratio))  # Penalty ranges from 2 to 4
        
        # Small reward for moving towards closest plant, bigger when hungry
        closest_plant = None
        min_dist = float('inf')
        for plant in self.plants:
            dist = math.sqrt((self.x - plant.x)**2 + (self.y - plant.y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_plant = plant
                
        if closest_plant:
            prev_dist = math.sqrt((prev_x - closest_plant.x)**2 + (prev_y - closest_plant.y)**2)
            curr_dist = math.sqrt((self.x - closest_plant.x)**2 + (self.y - closest_plant.y)**2)
            if curr_dist < prev_dist:
                hunger_ratio = 1 - (self.hunger / self.max_hunger)
                reward += 0.2 * (1 + hunger_ratio)  # Reward ranges from 0.2 to 0.4 based on hunger
        
        # Penalty for being very hungry (increases as hunger decreases)
        if self.hunger < self.max_hunger * 0.5:  # Below 50% hunger
            hunger_ratio = self.hunger / self.max_hunger
            reward -= 0.2 * (1 - hunger_ratio)  # Penalty ranges from 0 to 0.2
        
        # Get state and additional info
        state = self._get_state()
        info = {
            'wall_collision': wall_collision,
            'ate_plant': ate_plant,
            'hunger': self.hunger / self.max_hunger,
            'alive': alive,
            'reward': reward  # Add reward to info
        }
        
        return state, info
        
    def draw(self, surface=None):
        """Draw the game state"""
        if surface is None and not self.headless:
            surface = pygame.display.get_surface()
            
        # Clear game surface
        if not self.headless:
            self.game_surface.fill((50, 50, 50))
        
        # Draw plants
        for plant in self.plants:
            if not self.headless:
                plant.draw(self.game_surface)
        
        # Draw worm segments
        for i in range(1, len(self.positions)):
            pos = self.positions[i]
            prev_pos = self.positions[i-1]
            
            # Calculate angle between segments
            dx = pos[0] - prev_pos[0]
            dy = pos[1] - prev_pos[1]
            angle = math.atan2(dy, dx)
            
            # Draw segment
            if not self.headless:
                self._draw_segment(pos, angle, self.segment_width, self.body_colors[i-1])
        
        # Draw head (last segment) with special handling
        head_pos = self.positions[-1]
        if len(self.positions) > 1:
            prev_pos = self.positions[-2]
            dx = head_pos[0] - prev_pos[0]
            dy = head_pos[1] - prev_pos[1]
            head_angle = math.atan2(dy, dx)
        else:
            head_angle = 0
            
        # Draw head with slightly different appearance
        if not self.headless:
            self._draw_segment(head_pos, head_angle, self.segment_width * 1.2, (255, 100, 100), True)
        
        # Draw hunger meter
        if not self.headless:
            meter_width = 200
            meter_height = 20
            meter_x = 10
            meter_y = 10
            border_color = (200, 200, 200)
            
            # Calculate fill color based on hunger
            red = min(255, max(0, int(255 * (1 - self.hunger / self.max_hunger))))
            green = min(255, max(0, int(255 * self.hunger / self.max_hunger)))
            fill_color = (red, green, 0)
            
            # Draw border
            pygame.draw.rect(self.game_surface, border_color, (meter_x, meter_y, meter_width, meter_height), 2)
            
            # Draw fill
            fill_width = max(0, min(meter_width, int(meter_width * self.hunger / self.max_hunger)))
            pygame.draw.rect(self.game_surface, fill_color, (meter_x, meter_y, fill_width, meter_height))
        
        # Draw game surface to main surface
        if not self.headless:
            surface.blit(self.game_surface, (self.game_x_offset, self.game_y_offset))
        
        # Draw border around game area
        if not self.headless:
            pygame.draw.rect(surface, (100, 100, 100), 
                            (self.game_x_offset-2, self.game_y_offset-2, 
                             self.screen_width+4, self.screen_height+4), 2)
        
        if not self.headless:
            pygame.display.flip()
        
    def _draw_segment(self, pos, angle, width, color, is_head=False):
        """Draw a single body segment"""
        x, y = pos
        
        # Calculate rectangle points
        length = self.segment_length
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Calculate corners of the rectangle
        points = [
            (x - length/2 * cos_a - width/2 * sin_a,
             y - length/2 * sin_a + width/2 * cos_a),
            (x + length/2 * cos_a - width/2 * sin_a,
             y + length/2 * sin_a + width/2 * cos_a),
            (x + length/2 * cos_a + width/2 * sin_a,
             y + length/2 * sin_a - width/2 * cos_a),
            (x - length/2 * cos_a + width/2 * sin_a,
             y - length/2 * sin_a - width/2 * cos_a)
        ]
        
        if not self.headless:
            pygame.draw.polygon(self.game_surface, color, points)
        
        # Draw outline
        if not self.headless:
            pygame.draw.lines(self.game_surface, (0, 0, 0), True, points, 2)
        
        # Draw eyes if head
        if is_head and not self.headless:
            eye_radius = 3
            eye_x = x - 5 * cos_a - 5 * sin_a
            eye_y = y - 5 * sin_a + 5 * cos_a
            pygame.draw.circle(self.game_surface, (0, 0, 0), (int(eye_x), int(eye_y)), eye_radius)
            eye_x = x + 5 * cos_a - 5 * sin_a
            eye_y = y + 5 * sin_a + 5 * cos_a
            pygame.draw.circle(self.game_surface, (0, 0, 0), (int(eye_x), int(eye_y)), eye_radius)
    
    def _get_state(self):
        """Get the current state for the neural network"""
        # Normalize positions to [0,1] range
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
        
        # Base state
        state = [norm_x, norm_y, dist_right, dist_left, dist_bottom, dist_top, vel_x, vel_y]
        
        # Add closest plant information
        closest_dist = float('inf')
        closest_angle = 0
        plant_state = 0  # 0: no plant, 1: growing, 2: mature, 3: wilting
        
        for plant in self.plants:
            dx = plant.x - self.x
            dy = (plant.y + plant.size) - self.y  # Target the base of the plant
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_angle = math.atan2(dy, dx) - self.angle
                if plant.state == 'growing':
                    plant_state = 1
                elif plant.state == 'mature':
                    plant_state = 2
                else:  # wilting
                    plant_state = 3
                    
        # Normalize values
        closest_dist = min(1.0, closest_dist / self.screen_height)
        closest_angle = (closest_angle + math.pi) / (2 * math.pi)  # Normalize to [0,1]
        
        # Add plant and hunger info to state
        state.extend([
            closest_dist,
            closest_angle,
            plant_state / 3.0,  # Normalize state to [0,1]
            self.hunger / self.max_hunger
        ])
        
        return state

    def reset(self):
        """Reset the game state"""
        # Reset worm position to center
        self.x = self.width / 2
        self.y = self.height / 2
        self.angle = 0
        self.speed = 5
        
        # Reset segments
        self.num_segments = 20
        self.positions = [(self.x, self.y) for _ in range(self.num_segments)]
        
        # Reset colors
        self.body_colors = []
        for i in range(self.num_segments):
            green_val = int(150 - (100 * i / (self.max_segments - 1)))
            self.body_colors.append((70, green_val + 30, 20))
        
        # Reset hunger
        self.hunger = self.max_hunger
        
        # Clear and respawn plants
        self.plants = []
        for _ in range(self.max_plants):
            self.spawn_plant()
            
        # Get initial state
        return self._get_state()
        
    def update_hunger(self):
        """Update hunger and return whether worm is still alive"""
        self.hunger = max(0, self.hunger - self.hunger_rate)
        
        # Calculate shrinking probability based on hunger level
        hunger_ratio = self.hunger / self.max_hunger
        if hunger_ratio < 0.5:  # Start shrinking below 50% hunger
            # Shrinking probability increases as hunger decreases
            # At 50% hunger: 0.1% chance per frame
            # At 0% hunger: 5% chance per frame
            shrink_probability = 0.05 * (1 - (hunger_ratio * 2))
            
            if self.num_segments > self.min_segments and random.random() < shrink_probability:
                self.num_segments -= 1
                if len(self.positions) > self.num_segments:
                    self.positions.pop()  # Remove last segment
                    self.body_colors.pop()  # Remove its color
        
        # Die if no segments left or hunger is zero
        return self.hunger > 0 and self.num_segments > 0
        
class WormAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.batch_size = 2048
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9, verbose=True)
        
        # Try to load saved model
        try:
            checkpoint = torch.load('models/saved/worm_model.pth')
            if checkpoint['state_size'] == state_size:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Loaded training state. Epsilon: {self.epsilon:.4f}")
            else:
                print("Old model architecture detected, starting fresh")
        except:
            print("No saved model found, starting fresh")
            
        self.target_model.load_state_dict(self.model.state_dict())
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target model
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update model
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, episode):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size
        }, f'models/saved/worm_model.pth')
        print(f"Saved model state at episode {episode}")

# ML Agent setup
STATE_SIZE = 12  # position (2), velocity (2), distances to walls (4), closest plant (4)
ACTION_SIZE = 9  # 8 directions + no movement
agent = WormAgent(STATE_SIZE, ACTION_SIZE)

# Load the best model if in demo mode
if args.demo:
    agent.load_state = True
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

if __name__ == "__main__":
    game = WormGame(headless=False)  # Regular display mode for direct running
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Get current state
        state = game._get_state()
        
        # Get action from agent
        action = agent.act(state)
        
        # Execute action and get next state
        next_state, info = game.step(action)
        reward = info.get('reward', 0)
        done = not info['alive']
        
        # Remember experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train the agent
        if not args.demo:  # Only train if not in demo mode
            loss = agent.train()
            
            # Update target network periodically
            if steps_in_episode % 100 == 0:
                agent.update_target_model()
                
        # Draw everything
        game.draw()
        
        # Update display
        pygame.display.flip()
        
        if done:
            print("Game Over! Starting new episode...")
            game.reset()
            
        # Increment steps
        steps_in_episode += 1
        
        # Check if episode should end
        if steps_in_episode >= MAX_STEPS:
            # Save the model state every 5 episodes
            if episode % 5 == 0:
                agent.save(episode)
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
            
    pygame.quit()
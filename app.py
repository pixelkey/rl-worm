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
    def __init__(self):
        pygame.init()
        
        # Initialize display info
        display_info = pygame.display.Info()
        self.screen_width = min(800, display_info.current_w - 100)
        self.screen_height = min(600, display_info.current_h - 100)
        
        # Create game surface and screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Worm" + (" (Demo Mode)" if args.demo else ""))
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
        head_x, head_y = self.positions[0]
        for i, plant in enumerate(self.plants):
            dist = math.sqrt((head_x - plant.x)**2 + (head_y - (plant.y + plant.size))**2)
            if dist < self.head_size + plant.size:
                if plant.state == 'mature':
                    # Eat the plant
                    self.hunger = min(self.max_hunger, self.hunger + self.hunger_gain_from_plant)
                    if self.num_segments < self.max_segments:
                        self.num_segments += 1
                        self.body_colors.append(self.body_colors[-1])
                    self.plants.pop(i)
                    return True
        return False
        
    def update_hunger(self):
        self.hunger = max(0, self.hunger - self.hunger_rate)
        
        # Shrink if very hungry
        if self.hunger < self.max_hunger * 0.2 and self.num_segments > self.min_segments:
            if random.random() < 0.01:  # 1% chance per frame when hungry
                self.num_segments -= 1
                self.body_colors.pop()
                
        return self.hunger > 0 and self.num_segments >= self.min_segments
        
    def step(self, action):
        wall_collision = False
        old_head_pos = self.positions[0]
        
        # Update head position based on action
        if action < 8:  # Movement actions
            self.angle = action * math.pi / 4
            dx = math.cos(self.angle) * self.speed
            dy = math.sin(self.angle) * self.speed
            new_x = max(self.head_size, min(self.width - self.head_size, self.x + dx))
            new_y = max(self.head_size, min(self.height - self.head_size, self.y + dy))
            
            wall_collision = (new_x in (self.head_size, self.width - self.head_size) or 
                            new_y in (self.head_size, self.height - self.head_size))
            
            self.x, self.y = new_x, new_y
            
            # Calculate movement vector
            move_dx = self.x - old_head_pos[0]
            move_dy = self.y - old_head_pos[1]
            move_dist = math.sqrt(move_dx * move_dx + move_dy * move_dy)
            
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
        
        # Get state and additional info
        state = self._get_state()
        info = {
            'wall_collision': wall_collision,
            'ate_plant': ate_plant,
            'hunger': self.hunger / self.max_hunger,
            'alive': alive
        }
        
        return state, info
        
    def draw(self, surface=None):
        """Draw the game state"""
        if surface is None:
            surface = pygame.display.get_surface()
            
        # Clear game surface
        self.game_surface.fill((50, 50, 50))
        
        # Draw plants
        for plant in self.plants:
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
        self._draw_segment(head_pos, head_angle, self.segment_width * 1.2, (255, 100, 100), True)
        
        # Draw hunger meter
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
        surface.blit(self.game_surface, (self.game_x_offset, self.game_y_offset))
        
        # Draw border around game area
        pygame.draw.rect(surface, (100, 100, 100), 
                        (self.game_x_offset-2, self.game_y_offset-2, 
                         self.screen_width+4, self.screen_height+4), 2)
        
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
        
        pygame.draw.polygon(self.game_surface, color, points)
        
        # Draw outline
        pygame.draw.lines(self.game_surface, (0, 0, 0), True, points, 2)
        
        # Draw eyes if head
        if is_head:
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
        self.x = self.width // 2
        self.y = self.height // 2
        self.angle = 0
        self.speed = 5
        
        # Initialize body segments in a straight line behind the head
        self.positions = []
        dx = -math.cos(self.angle) * (self.segment_width * self.spacing)
        dy = -math.sin(self.angle) * (self.segment_width * self.spacing)
        
        for i in range(self.num_segments):
            pos_x = self.x + dx * i
            pos_y = self.y + dy * i
            self.positions.append((pos_x, pos_y))
        
        return self._get_state()

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

game = WormGame()

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
        print("Game Over! Resetting...")
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
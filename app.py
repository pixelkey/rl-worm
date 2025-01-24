import pygame
import random
import math
import numpy as np
import torch
import time
import os
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
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
        self.size = int(game_size/40)  # Plant size scales with game
        self.max_lifetime = 500  # How long the plant lives
        self.lifetime = self.max_lifetime
        self.state = 'growing'  # growing, mature, wilting
        self.color = (0, 180, 0)  # Start with bright green
        
    def update(self):
        """Update plant state based on lifetime"""
        self.lifetime -= 1
        life_ratio = self.lifetime / self.max_lifetime
        
        # Update state and color based on lifetime
        if life_ratio > 0.7:  # First 30% of life
            self.state = 'growing'
            green = int(180 * (life_ratio - 0.7) / 0.3)  # Fade in from black
            self.color = (0, green, 0)
        elif life_ratio > 0.3:  # Middle 40% of life
            self.state = 'mature'
            self.color = (0, 180, 0)  # Bright green
        else:  # Last 30% of life
            self.state = 'wilting'
            green = int(180 * life_ratio / 0.3)  # Fade to brown
            self.color = (100, green, 0)
            
        return self.lifetime > 0
        
    def draw(self, surface):
        """Draw the plant"""
        # Draw stem
        stem_height = self.size * 2
        stem_width = max(2, self.size // 4)
        stem_rect = pygame.Rect(self.x - stem_width//2, 
                              self.y - stem_height//2,
                              stem_width, stem_height)
        pygame.draw.rect(surface, self.color, stem_rect)
        
        # Draw leaves
        leaf_size = self.size
        leaf_points = [
            # Left leaf
            [(self.x - leaf_size, self.y),
             (self.x, self.y - leaf_size//2),
             (self.x, self.y + leaf_size//2)],
            # Right leaf
            [(self.x + leaf_size, self.y),
             (self.x, self.y - leaf_size//2),
             (self.x, self.y + leaf_size//2)]
        ]
        
        for points in leaf_points:
            pygame.draw.polygon(surface, self.color, points)
            # Draw leaf veins
            pygame.draw.line(surface, (0, min(255, self.color[1] + 30), 0),
                           points[1], points[0], max(1, stem_width//2))
            pygame.draw.line(surface, (0, min(255, self.color[1] + 30), 0),
                           points[2], points[0], max(1, stem_width//2))

class WormGame:
    def __init__(self, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            
        pygame.init()
        self.headless = headless
        
        # Set dimensions based on mode
        if headless:
            # Use smaller fixed dimensions for headless mode
            self.screen_width = 800
            self.screen_height = 600
            self.game_width = 800
            self.game_height = 600
        else:
            # Get display info for window sizing
            display_info = pygame.display.Info()
            screen_height = min(1600, display_info.current_h - 100)  # Doubled from 800
            self.screen_width = int(screen_height * 1.2)  # 20% wider than height
            self.screen_height = screen_height
            
            # Game area is 80% of screen height
            self.game_width = int(self.screen_height * 0.8)
            self.game_height = int(self.screen_height * 0.8)
        
        # Create game surface and screen based on mode
        if not headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("AI Worm" + (" (Demo Mode)" if args.demo else ""))
        else:
            pygame.display.set_mode((1, 1))  # Minimal display for headless mode
            
        # Create game surface for the actual play area
        self.game_surface = pygame.Surface((self.game_width, self.game_height))
        
        # Calculate game area offset to center it
        self.game_x_offset = (self.screen_width - self.game_width) // 2
        self.game_y_offset = (self.screen_height - self.game_height) // 2
        
        # Movement properties
        self.angle = 0  # Current angle
        self.target_angle = 0  # Target angle
        self.angular_speed = 0.1  # How fast we can turn (reduced from 0.15)
        self.speed = 5.0  # Movement speed
        self.prev_action = 4  # Previous action (start with no movement)
        
        # Worm properties - scale with game area
        self.segment_length = int(self.game_height/20)  # Spacing between segments
        self.segment_width = int(self.game_height/25)   # Size of body segments
        self.num_segments = 20
        self.max_segments = 30
        self.min_segments = 5
        self.head_size = int(self.game_height/20)      # Size of head
        self.spacing = 0.8  # Spacing between segments
        
        # Eye properties
        self.eye_size = int(self.head_size * 0.25)
        self.eye_offset = int(self.head_size * 0.3)
        
        # Expression properties
        self.expression = 0  # -1 for frown, 0 for neutral, 1 for smile
        self.expression_time = 0  # Time when expression was set
        self.expression_duration = 2.0  # Duration in seconds before returning to neutral
        self.last_time = time.time()  # For tracking time between frames
        
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
        self.head_color = (150, 50, 50)  # Reddish
        self.eye_color = (255, 255, 255)  # White
        self.pupil_color = (0, 0, 0)  # Black
        self.body_colors = []
        for i in range(self.max_segments):
            green_val = int(150 - (100 * i / self.max_segments))
            self.body_colors.append((70, green_val + 30, 20))
        
        # Game boundaries (use game area dimensions)
        self.width = self.game_width
        self.height = self.game_height
        
        # Initialize worm
        self.reset()

    def spawn_plant(self):
        if len(self.plants) < self.max_plants and random.random() < self.plant_spawn_rate:
            margin = self.game_height // 10
            x = random.randint(margin, self.game_width - margin)
            y = random.randint(margin, self.game_height - margin)
            
            # Check if too close to other plants
            too_close = False
            for plant in self.plants:
                dist = math.sqrt((x - plant.x)**2 + (y - plant.y)**2)
                if dist < self.game_height // 8:
                    too_close = True
                    break
                    
            if not too_close:
                self.plants.append(Plant(x, y, self.game_height))
                
    def update_plants(self):
        # Update existing plants
        self.plants = [plant for plant in self.plants if plant.update()]
        
        # Try to spawn new plant
        self.spawn_plant()
        
    def check_plant_collision(self):
        """Check for collisions with plants and handle eating"""
        head_rect = pygame.Rect(self.positions[0][0] - self.head_size/2, 
                              self.positions[0][1] - self.head_size/2,
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
        # Convert action to target angle
        self.target_angle = action * (2 * math.pi / 8)  # Convert to radians (8 possible directions)
        
        # Smoothly interpolate current angle towards target angle
        angle_diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        if abs(angle_diff) > self.angular_speed:
            if angle_diff > 0:
                self.angle += self.angular_speed
            else:
                self.angle -= self.angular_speed
        else:
            self.angle = self.target_angle
            
        # Normalize angle
        self.angle = self.angle % (2 * math.pi)
        
        # Previous position for movement calculations
        prev_x = self.x
        prev_y = self.y
        
        # Calculate movement
        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed
        
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
            
        # Always update head position
        if len(self.positions) == 0:
            self.positions.append((self.x, self.y))
        else:
            # Only update if position has changed
            current_head = self.positions[0]
            if (self.x, self.y) != current_head:
                self.positions.insert(0, (self.x, self.y))
                # Remove last position if we have too many
                if len(self.positions) > self.num_segments:
                    self.positions.pop()
        
        # Ensure we maintain the correct number of body segments
        while len(self.positions) < self.num_segments:
            # If we don't have enough positions, duplicate the last one
            last_pos = self.positions[-1] if self.positions else (self.x, self.y)
            self.positions.append(last_pos)
        
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
            base_reward = 10.0 * (1 + hunger_ratio)  # Base reward ranges from 10 to 20 based on hunger
            reward += base_reward
            
            # Extra reward for growing
            growth_reward = 0
            if self.num_segments > len(self.positions) - 1:  # If we grew
                growth_reward = 5.0  # Bonus for growing
                reward += growth_reward
            
            # Set happy expression scaled by total reward (normalize to 0-1 range, assuming max reward ~25)
            total_reward = base_reward + growth_reward
            self.expression = min(1.0, total_reward / 25.0)
            self.expression_time = time.time()
                
        # Penalty for hitting walls
        if wall_collision:
            wall_penalty = -1.0
            reward += wall_penalty
            # Set sad expression scaled by penalty (normalize to -1 to 0 range)
            self.expression = max(-1.0, wall_penalty / -2.0)  # -2.0 is considered max penalty
            self.expression_time = time.time()
            
        # Penalty for sharp turns
        action_diff = abs(action - self.prev_action)
        if action_diff > 4:  # If turning more than 180 degrees
            action_diff = 8 - action_diff  # Use smaller angle
        if action_diff > 2:  # Penalize turns sharper than 90 degrees
            turn_penalty = -0.5 * (action_diff - 2)  # Progressive penalty for sharper turns
            reward += turn_penalty
            # Set sad expression scaled by turn penalty
            self.expression = max(-1.0, turn_penalty / -2.0)  # Scale to -1 to 0 range
            self.expression_time = time.time()
        
        self.prev_action = action
        
        # Get state and additional info
        state = self._get_state()
        info = {
            'reward': reward,
            'alive': alive,
            'ate_plant': ate_plant,
            'wall_collision': wall_collision
        }
        
        return state, info
        
    def draw(self, surface=None):
        """Draw the game state"""
        if surface is None and not self.headless:
            surface = pygame.display.get_surface()
            
        # Clear game surface with a dark background
        if not self.headless:
            self.game_surface.fill((30, 30, 30))  # Darker background
            
            # Draw grid lines for visual reference
            grid_spacing = self.game_height // 10
            for i in range(0, self.game_width + grid_spacing, grid_spacing):
                pygame.draw.line(self.game_surface, (40, 40, 40), (i, 0), (i, self.game_height))
            for i in range(0, self.game_height + grid_spacing, grid_spacing):
                pygame.draw.line(self.game_surface, (40, 40, 40), (0, i), (self.game_width, i))
        
        # Draw plants
        for plant in self.plants:
            if not self.headless:
                plant.draw(self.game_surface)
        
        # Draw worm segments from tail to head
        for i in range(len(self.positions)-1, 0, -1):
            pos = self.positions[i]
            next_pos = self.positions[i-1]  # Next segment towards head
            
            # Calculate angle between segments
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            angle = math.atan2(dy, dx)
            
            # Draw segment
            if not self.headless:
                self._draw_segment(pos, angle, self.segment_width, self.body_colors[i])
        
        # Draw head (first segment)
        head_pos = self.positions[0]
        if len(self.positions) > 1:
            next_pos = self.positions[1]
            dx = head_pos[0] - next_pos[0]  # Reversed to point outward
            dy = head_pos[1] - next_pos[1]
            head_angle = math.atan2(dy, dx)
        else:
            head_angle = self.angle
            
        # Draw head with slightly different appearance
        if not self.headless:
            self._draw_segment(head_pos, head_angle, self.segment_width * 1.2, self.head_color, True)
        
        # Draw hunger meter
        if not self.headless:
            meter_width = self.game_width // 4
            meter_height = self.game_height // 20
            meter_x = self.game_width // 20
            meter_y = self.game_height // 20
            border_color = (200, 200, 200)
            
            # Calculate fill color based on hunger
            hunger_ratio = self.hunger / self.max_hunger
            if hunger_ratio > 0.6:
                fill_color = (0, 255, 0)  # Green when full
            elif hunger_ratio > 0.3:
                fill_color = (255, 255, 0)  # Yellow when half
            else:
                fill_color = (255, 0, 0)  # Red when low
            
            # Draw border
            pygame.draw.rect(self.game_surface, border_color, 
                           (meter_x, meter_y, meter_width, meter_height), 2)
            
            # Draw fill
            fill_width = max(0, min(meter_width-4, int((meter_width-4) * hunger_ratio)))
            pygame.draw.rect(self.game_surface, fill_color,
                           (meter_x+2, meter_y+2, fill_width, meter_height-4))
            
            # Draw length indicator
            length_text = f"Length: {self.num_segments}"
            font = pygame.font.Font(None, self.game_height // 20)
            text_surface = font.render(length_text, True, (200, 200, 200))
            text_rect = text_surface.get_rect()
            text_rect.topleft = (meter_x, meter_y + meter_height + 5)
            self.game_surface.blit(text_surface, text_rect)
        
        # Draw game area border
        if not self.headless:
            # Clear screen
            surface.fill((20, 20, 20))
            
            # Draw game surface onto main screen with offset
            surface.blit(self.game_surface, (self.game_x_offset, self.game_y_offset))
            
            # Draw border around game area
            pygame.draw.rect(surface, (100, 100, 100), 
                           (self.game_x_offset-2, self.game_y_offset-2, 
                            self.game_width+4, self.game_height+4), 2)
        
        if not self.headless:
            pygame.display.flip()
    
    def _draw_segment(self, pos, angle, width, color, is_head=False):
        """Draw a single body segment"""
        x, y = pos
        
        # Draw main body circle
        pygame.draw.circle(self.game_surface, color, (int(x), int(y)), width)
        
        if is_head:
            # Calculate eye positions (rotate around head center)
            eye_offset = self.head_size * 0.3  # Distance from center
            
            # Calculate eye positions relative to movement direction
            eye_y_offset = self.head_size * 0.1  # Eyes slightly up
            
            # Base eye positions (before rotation)
            base_left_x = -eye_offset
            base_left_y = -eye_y_offset
            base_right_x = eye_offset
            base_right_y = -eye_y_offset
            
            # Rotate eye positions based on worm's angle
            left_eye_x = x + (base_left_x * math.cos(angle) - base_left_y * math.sin(angle))
            left_eye_y = y + (base_left_x * math.sin(angle) + base_left_y * math.cos(angle))
            right_eye_x = x + (base_right_x * math.cos(angle) - base_right_y * math.sin(angle))
            right_eye_y = y + (base_right_x * math.sin(angle) + base_right_y * math.cos(angle))
            
            # Draw eyes (white part)
            pygame.draw.circle(self.game_surface, self.eye_color, 
                            (int(left_eye_x), int(left_eye_y)), self.eye_size)
            pygame.draw.circle(self.game_surface, self.eye_color,
                            (int(right_eye_x), int(right_eye_y)), self.eye_size)
            
            # Draw pupils (black part)
            pupil_size = self.eye_size // 2
            pygame.draw.circle(self.game_surface, self.pupil_color,
                            (int(left_eye_x), int(left_eye_y)), pupil_size)
            pygame.draw.circle(self.game_surface, self.pupil_color,
                            (int(right_eye_x), int(right_eye_y)), pupil_size)
            
            # Update expression timing
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Calculate expression interpolation
            if self.expression != 0:
                time_since_expression = current_time - self.expression_time
                if time_since_expression < self.expression_duration:
                    # Smoothly interpolate back to neutral
                    t = time_since_expression / self.expression_duration
                    current_expression = self.expression * (1 - t)
                else:
                    self.expression = 0
                    current_expression = 0
            else:
                current_expression = 0
            
            # Draw mouth (centered and below eyes)
            mouth_width = self.head_size * 0.7  # Bigger mouth
            mouth_height = self.head_size * 0.3  # Increased height for more expressive curves
            mouth_y_offset = self.head_size * 0.2  # Move down from center
            
            # Base mouth points (before rotation)
            base_left_x = -mouth_width/2
            base_left_y = mouth_y_offset
            base_right_x = mouth_width/2
            base_right_y = mouth_y_offset
            
            # Rotate mouth points based on worm's angle
            left_x = x + (base_left_x * math.cos(angle) - base_left_y * math.sin(angle))
            left_y = y + (base_left_x * math.sin(angle) + base_left_y * math.cos(angle))
            right_x = x + (base_right_x * math.cos(angle) - base_right_y * math.sin(angle))
            right_y = y + (base_right_x * math.sin(angle) + base_right_y * math.cos(angle))
            
            # Calculate control point for curved mouth
            curve_height = mouth_height * current_expression
            base_control_x = 0
            base_control_y = mouth_y_offset + curve_height
            
            control_x = x + (base_control_x * math.cos(angle) - base_control_y * math.sin(angle))
            control_y = y + (base_control_x * math.sin(angle) + base_control_y * math.cos(angle))
            
            # Draw curved mouth using quadratic Bezier
            points = [(int(left_x), int(left_y)), 
                     (int(control_x), int(control_y)),
                     (int(right_x), int(right_y))]
            pygame.draw.lines(self.game_surface, self.pupil_color, False, points, 2)
    
    def _get_state(self):
        """Get the current state for the neural network"""
        # Calculate distances to walls
        wall_dists = [
            self.x,  # Distance to left wall
            self.width - self.x,  # Distance to right wall
            self.y,  # Distance to top wall
            self.height - self.y  # Distance to bottom wall
        ]
        
        # Calculate velocity
        if len(self.positions) > 1:
            prev_x, prev_y = self.positions[1]
            vel_x = (self.x - prev_x) / self.segment_length
            vel_y = (self.y - prev_y) / self.segment_length
        else:
            vel_x = vel_y = 0
        
        # Find closest plant
        closest_dist = float('inf')
        closest_angle = 0
        plant_state = 0
        
        for plant in self.plants:
            dx = plant.x - self.x
            dy = plant.y - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < closest_dist:
                closest_dist = dist
                # Calculate angle to plant relative to worm's current angle
                plant_angle = math.atan2(dy, dx)
                closest_angle = (plant_angle - self.angle + math.pi) % (2*math.pi) - math.pi
                
                # Get plant state
                if plant.state == 'growing':
                    plant_state = 1
                elif plant.state == 'mature':
                    plant_state = 2
                else:  # wilting
                    plant_state = 3
        
        # Normalize values
        closest_dist = min(1.0, closest_dist / self.game_height)
        closest_angle = (closest_angle + math.pi) / (2 * math.pi)  # Normalize to [0,1]
        plant_state = plant_state / 3.0  # Normalize to [0,1]
        
        # Calculate angular velocity (difference between current and target angle)
        angle_diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        angular_vel = angle_diff / math.pi  # Normalize to [-1,1]
        
        # Normalize current angle
        norm_angle = (self.angle % (2 * math.pi)) / (2 * math.pi)  # Normalize to [0,1]
        
        # Combine all state components
        state = [
            self.x / self.width,  # Normalized position
            self.y / self.height,
            vel_x,  # Already normalized by segment_length
            vel_y,
            norm_angle,  # Current angle [0,1]
            angular_vel,  # Angular velocity [-1,1]
            closest_dist,  # Distance to closest plant [0,1]
            closest_angle,  # Angle to closest plant [0,1]
            plant_state,  # State of closest plant [0,1]
            wall_dists[0] / self.width,  # Normalized wall distances
            wall_dists[1] / self.width,
            wall_dists[2] / self.height,
            wall_dists[3] / self.height,
            self.hunger / self.max_hunger  # Normalized hunger [0,1]
        ]
        
        return state

    def reset(self):
        """Reset the game state"""
        # Reset worm position to center
        self.x = self.width / 2
        self.y = self.height / 2
        self.angle = 0
        self.target_angle = 0
        self.prev_action = 4  # Reset to no movement
        
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
STATE_SIZE = 14  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), walls (4), hunger (1)
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
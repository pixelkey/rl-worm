import pygame
import random
import math
import numpy as np
import torch
import os
import time
import argparse
from collections import deque
from plant import Plant  # Import the Plant class
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from analytics.metrics import WormAnalytics

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the intelligent worm simulation')
parser.add_argument('--demo', action='store_true', help='Run in demo mode using the best trained model')
args = parser.parse_args()

class WormGame:
    def __init__(self, headless=False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            # Force OpenGL to use NVIDIA driver
            os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
            os.environ["SDL_VIDEODRIVER"] = "x11"
            
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
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.SWSURFACE)
            pygame.display.set_caption("AI Worm" + (" (Demo Mode)" if args.demo else ""))
        else:
            pygame.display.set_mode((1, 1))  # Minimal display for headless mode
            
        # Create game surface for the actual play area
        self.game_surface = pygame.Surface((self.game_width, self.game_height))
        
        # Calculate game area offset to center it
        self.game_x_offset = (self.screen_width - self.game_width) // 2
        self.game_y_offset = (self.screen_height - self.game_height) // 2
        
        # Level and step tracking
        self.level = 1
        self.episode_reward = 0.0
        self.steps_in_level = 0
        self.min_steps = 2000  # Start with 2000 steps like training
        self.max_steps = 5000  # Maximum steps like training
        self.steps_increment = 100  # How many steps to add when leveling up
        self.steps_for_level = self.min_steps  # Current level's step requirement
        
        # Add level and episode tracking
        self.base_steps_for_level = 1000
        self.level_step_increase = 0.2  # 20% increase per level
        
        # Font for displaying text
        if not headless:
            self.font = pygame.font.Font(None, 36)
        
        # Movement properties
        self.angle = 0  # Current angle
        self.target_angle = 0  # Target angle
        self.angular_speed = 0.05  # Reduced from 0.1 for smoother turning
        self.speed = 4.0  # Slightly reduced from 5.0
        self.prev_action = 4  # Previous action (start with no movement)
        
        # Worm properties - scale with game area
        self.segment_length = int(self.game_height/20)  # Spacing between segments
        self.segment_width = int(self.game_height/25)   # Size of body segments
        self.num_segments = 2  # Starting number of segments
        self.max_segments = 30
        self.min_segments = 2
        self.head_size = int(self.game_height/20)      # Size of head
        self.segment_spacing = self.head_size * 1.2  # Fixed spacing between segments
        
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
        self.base_hunger_rate = 0.1
        self.current_hunger_rate = self.base_hunger_rate
        self.hunger_gain_from_plant = 400  # Increased from 300
        self.shrink_hunger_threshold = 0.4  # Start shrinking at 40% hunger instead of 50%
        self.shrink_cooldown = 60  # Frames between shrinks
        self.shrink_timer = 0  # Counter for shrink cooldown
        
        # Plant management
        self.min_plants = 2
        self.max_plants = 8
        self.target_plants = random.randint(self.min_plants, self.max_plants)
        self.plant_spawn_chance = 0.02  # 2% chance per frame to spawn a new plant
        self.plant_spawn_cooldown = 0
        self.plant_spawn_cooldown_max = 60  # Minimum frames between spawns
        
        # Plant mechanics
        self.plants = []
        self.base_plant_spawn_chance = 0.01  # Reduced from 0.02
        self.current_plant_spawn_chance = self.base_plant_spawn_chance
        self.base_max_plants = 3  # Reduced from 5
        self.current_max_plants = self.base_max_plants
        
        # Colors
        self.head_color = (150, 50, 50)  # Reddish
        self.eye_color = (255, 255, 255)  # White
        self.pupil_color = (0, 0, 0)  # Black
        self.body_colors = []
        for i in range(self.max_segments):
            green_val = int(150 - (100 * i / self.max_segments))
            self.body_colors.append((70, green_val + 30, 20))
        self.wall_color = (70, 70, 70)  # Dark gray
        self.spike_color = (70, 70, 70)   # Darker gray for spikes
        
        # Game boundaries (use game area dimensions)
        self.width = self.game_width
        self.height = self.game_height
        
        # Generate rocky walls once at initialization
        self.wall_points = 100  # More points for finer detail
        self.wall_color = (70, 70, 70)  # Dark gray
        margin = int(self.head_size * 0.4)  # Reduced margin to 40% of original
        base_jitter = self.head_size * 0.4  # Halved from 0.8 to 0.4
        
        def get_jagged_wall_points(start, end, num_points):
            points = []
            # Get wall direction
            is_vertical = start[0] == end[0]
            wall_length = end[1] - start[1] if is_vertical else end[0] - start[0]
            
            # Generate base points along the wall
            for i in range(num_points):
                t = i / (num_points - 1)
                # Base position along wall
                base_x = start[0] + (0 if is_vertical else t * wall_length)
                base_y = start[1] + (t * wall_length if is_vertical else 0)
                
                # Calculate jitter based on position
                # More jitter in middle, less at corners
                edge_factor = min(t, 1-t) * 3  # Reduced from 4 to 3 for smoother transitions
                jitter_scale = base_jitter * (0.5 + edge_factor)
                
                # Add random jitter inward
                if is_vertical:
                    if start[0] == margin:  # Left wall
                        jit_x = random.random() * jitter_scale
                    else:  # Right wall
                        jit_x = -random.random() * jitter_scale
                    # Add some vertical displacement for more jaggedness
                    jit_y = (random.random() - 0.5) * jitter_scale * 0.25  # Reduced from 0.5 to 0.25
                else:
                    if start[1] == margin:  # Top wall
                        jit_y = random.random() * jitter_scale
                    else:  # Bottom wall
                        jit_y = -random.random() * jitter_scale
                    # Add some horizontal displacement for more jaggedness
                    jit_x = (random.random() - 0.5) * jitter_scale * 0.25  # Reduced from 0.5 to 0.25
                
                # Add sharp spikes randomly (reduced frequency and intensity)
                if random.random() < 0.1:  # Reduced from 0.2 to 0.1 (10% chance)
                    if is_vertical:
                        jit_x *= 1.5  # Reduced from 2.0 to 1.5
                    else:
                        jit_y *= 1.5  # Reduced from 2.0 to 1.5
                
                # Ensure points don't go outside the frame
                if is_vertical:
                    if start[0] == margin:  # Left wall
                        base_x = max(2, base_x + jit_x)  # Keep 2 pixels from edge
                    else:  # Right wall
                        base_x = min(self.width - 2, base_x + jit_x)
                    base_y = base_y + jit_y
                else:
                    if start[1] == margin:  # Top wall
                        base_y = max(2, base_y + jit_y)
                    else:  # Bottom wall
                        base_y = min(self.height - 2, base_y + jit_y)
                    base_x = base_x + jit_x
                
                points.append((base_x, base_y))
            
            # Add intermediate points for smoother jagged edges
            smoothed_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                smoothed_points.append(p1)
                
                # Add 2 intermediate points between each pair of main points
                for j in range(1, 3):
                    t = j / 3
                    # Linear interpolation with some random displacement
                    mid_x = p1[0] + t * (p2[0] - p1[0]) + (random.random() - 0.5) * base_jitter * 0.3
                    mid_y = p1[1] + t * (p2[1] - p1[1]) + (random.random() - 0.5) * base_jitter * 0.3
                    smoothed_points.append((mid_x, mid_y))
            
            smoothed_points.append(points[-1])
            return smoothed_points
        
        # Generate all wall points once
        self.left_wall = get_jagged_wall_points((margin, margin), (margin, self.height-margin), self.wall_points)
        self.right_wall = get_jagged_wall_points((self.width-margin, margin), (self.width-margin, self.height-margin), self.wall_points)
        self.top_wall = get_jagged_wall_points((margin, margin), (self.width-margin, margin), self.wall_points)
        self.bottom_wall = get_jagged_wall_points((margin, self.height-margin), (self.width-margin, self.height-margin), self.wall_points)
        
        # Reward/Penalty constants
        self.REWARD_FOOD_BASE = 100.0  # Doubled from 50.0
        self.REWARD_FOOD_HUNGER_SCALE = 5.0  # Increased from 3.0
        self.REWARD_GROWTH = 30.0  # Doubled from 15.0
        self.REWARD_SMOOTH_MOVEMENT = 0.2  # Increased from 0.1
        self.REWARD_EXPLORATION = 0.05  # Increased from 0.01
        
        self.PENALTY_WALL = -10.0  # Stronger wall penalty to discourage wall-hugging
        self.PENALTY_WALL_STAY = -5.0  # New penalty for staying near walls
        self.PENALTY_SHARP_TURN = -0.5  # Reduced from -1.0
        self.PENALTY_STARVATION_BASE = -0.1  # Reduced from -0.2
        self.PENALTY_DIRECTION_CHANGE = -0.1  # Reduced from -0.2
        self.PENALTY_SHRINK = -20.0  # New penalty for losing a segment
        
        # Expression scaling
        self.EXPRESSION_SCALE = 2.5  # Divide rewards/penalties by this to get expression
        
        # Initialize worm
        self.reset()

    def spawn_plant(self):
        """Try to spawn a new plant if conditions are met"""
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
            return True
        return False

    def update_plants(self):
        """Update all plants and handle spawning"""
        # Update existing plants
        for plant in self.plants[:]:  # Use slice copy to safely modify during iteration
            if not plant.update():
                self.plants.remove(plant)
                
        # Occasionally change target number of plants
        if random.random() < 0.001:  # 0.1% chance per frame
            self.target_plants = random.randint(self.min_plants, self.max_plants)
        
        # Handle plant spawning
        if self.plant_spawn_cooldown > 0:
            self.plant_spawn_cooldown -= 1
        
        # Adjust spawn chance based on current vs target number of plants
        current_plants = len(self.plants)
        if current_plants < self.target_plants:
            # Increase spawn chance when below target
            effective_spawn_chance = self.plant_spawn_chance * (1 + (self.target_plants - current_plants) * 0.5)
        else:
            # Decrease spawn chance when at or above target
            effective_spawn_chance = self.plant_spawn_chance * 0.5
        
        # Try to spawn a new plant
        if (current_plants < self.max_plants and 
            self.plant_spawn_cooldown <= 0 and 
            random.random() < effective_spawn_chance):
            
            self.spawn_plant()
            self.plant_spawn_cooldown = self.plant_spawn_cooldown_max

    def check_plant_collision(self):
        """Check for collisions with plants and handle eating"""
        # Only check head collisions for eating
        head_x, head_y = self.positions[0]
        
        # Create a collision box that matches the head's actual size
        head_rect = pygame.Rect(
            head_x - self.head_size/2,  # Center the box on the head
            head_y - self.head_size/2,
            self.head_size,            # Use actual head size
            self.head_size
        )
        
        # Check each plant
        for plant in self.plants[:]:  # Use slice copy to safely modify during iteration
            plant_rect = plant.get_bounding_box()
            
            # Get plant's nutritional value (0.0 to 1.0)
            nutrition = plant.get_nutritional_value()
            
            if head_rect.colliderect(plant_rect):
                # Full hunger gain based on nutrition (no division)
                hunger_gain = int(self.hunger_gain_from_plant * nutrition)
                
                # Apply the gains
                self.hunger = min(self.max_hunger, self.hunger + hunger_gain)
                
                # Only grow from healthy plants (nutrition > 0.5)
                if self.num_segments < self.max_segments and nutrition > 0.5:
                    self.num_segments += 1
                    # Add new segment at the end
                    last_pos = self.positions[-1]
                    self.positions.append(last_pos)
                    # Add new color for the segment
                    green_val = int(150 - (100 * (self.num_segments-1) / (self.max_segments - 1)))
                    self.body_colors.append((70, green_val + 30, 20))
                    # Show happy expression for growing
                    self.expression = 1
                    self.expression_time = time.time()
                else:
                    # Expression based on nutritional value
                    self.expression = min(1.0, nutrition)  # Better nutrition = happier expression
                    self.expression_time = time.time()
                
                # Remove the plant after eating it
                self.plants.remove(plant)
                return True
                
        return False

    def update_hunger(self):
        """Update hunger and return whether worm is still alive"""
        old_hunger = self.hunger
        old_segments = self.num_segments
        self.hunger = max(0, self.hunger - self.current_hunger_rate)
        
        # Update shrink timer
        if self.shrink_timer > 0:
            self.shrink_timer -= 1
        
        # Calculate shrinking based on hunger level
        hunger_ratio = self.hunger / self.max_hunger
        if hunger_ratio < self.shrink_hunger_threshold and self.shrink_timer == 0:
            if self.num_segments > self.min_segments:
                # Show sad expression when shrinking
                self.expression = -1
                self.expression_time = time.time()
                # Remove last segment
                self.num_segments -= 1
                if len(self.positions) > self.num_segments:
                    self.positions.pop()
                    self.body_colors.pop()
                # Set cooldown
                self.shrink_timer = self.shrink_cooldown
                # Return shrink occurred
                return self.hunger > 0 and self.num_segments > 0, True
            elif old_hunger > 0 and self.hunger == 0:
                # Show very sad expression when at minimum size and starving
                self.expression = -1
                self.expression_time = time.time()
        
        # Die if no segments left or hunger is zero
        return self.hunger > 0 and self.num_segments > 0, False
        
    def step(self, action):
        """Execute one time step within the environment"""
        # Update difficulty based on worm length
        difficulty_factor = (len(self.positions) - self.min_segments) / 10  # Every 10 segments above min increases difficulty
        self.current_hunger_rate = self.base_hunger_rate * (1 + difficulty_factor * 0.1)  # 10% faster hunger per difficulty level
        self.current_plant_spawn_chance = self.base_plant_spawn_chance / (1 + difficulty_factor * 0.1)  # 10% fewer plants per difficulty level
        self.current_max_plants = max(2, self.base_max_plants - int(difficulty_factor))  # Reduce max plants but keep at least 2
        
        # Store previous state for reward calculation
        prev_hunger = self.hunger
        prev_num_segments = len(self.positions)
        
        # Execute action and update target angle
        if action < 8:  # Directional movement
            self.target_angle = (action / 8) * 2 * math.pi
            # Smoothly interpolate current angle towards target
            angle_diff = (self.target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += np.clip(angle_diff, -self.angular_speed, self.angular_speed)
            
            # Calculate movement based on current angle (not target)
            dx = math.cos(self.angle) * self.speed
            dy = math.sin(self.angle) * self.speed
        else:  # No movement
            dx, dy = 0, 0
        
        # Update head position
        new_head_x = self.positions[0][0] + dx
        new_head_y = self.positions[0][1] + dy
        
        # Check wall collisions and constrain position
        wall_collision = False
        margin = self.head_size
        if new_head_x < margin:
            new_head_x = margin
            wall_collision = True
            self.angle = math.pi - self.angle + random.uniform(-0.2, 0.2)  # Bounce with slight randomness
        elif new_head_x > self.width - margin:
            new_head_x = self.width - margin
            wall_collision = True
            self.angle = math.pi - self.angle + random.uniform(-0.2, 0.2)  # Bounce with slight randomness
        if new_head_y < margin:
            new_head_y = margin
            wall_collision = True
            self.angle = -self.angle + random.uniform(-0.2, 0.2)  # Bounce with slight randomness
        elif new_head_y > self.height - margin:
            new_head_y = self.height - margin
            wall_collision = True
            self.angle = -self.angle + random.uniform(-0.2, 0.2)  # Bounce with slight randomness
            
        # Update head position
        self.positions[0] = (new_head_x, new_head_y)
        
        # Update body segment positions with fixed spacing
        for i in range(1, self.num_segments):
            # Get direction to previous segment
            prev_x, prev_y = self.positions[i-1]
            if i < len(self.positions):
                curr_x, curr_y = self.positions[i]
            else:
                # If this is a new segment, place it behind the previous one
                angle = self.angle + math.pi  # Opposite direction of movement
                curr_x = prev_x - math.cos(angle) * self.segment_spacing
                curr_y = prev_y - math.sin(angle) * self.segment_spacing
            
            # Calculate direction from current to previous segment
            dx = prev_x - curr_x
            dy = prev_y - curr_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Normalize direction
                dx /= dist
                dy /= dist
                
                # Position segment at fixed distance from previous segment
                new_x = prev_x - dx * self.segment_spacing
                new_y = prev_y - dy * self.segment_spacing
                
                # Update position
                if i < len(self.positions):
                    self.positions[i] = (new_x, new_y)
                else:
                    self.positions.append((new_x, new_y))
            
        # Ensure we have exactly num_segments positions
        while len(self.positions) > self.num_segments:
            self.positions.pop()
        
        # Update plants and check collisions
        self.update_plants()
        ate_plant = self.check_plant_collision()
        
        # Update hunger and check if shrink occurred
        alive, did_shrink = self.update_hunger()
        
        # Calculate reward using Maslow's hierarchy
        reward = 0
        
        # 1. Physiological Needs (Survival) - Highest Priority
        if ate_plant:
            hunger_ratio = 1 - (self.hunger / self.max_hunger)
            base_reward = self.REWARD_FOOD_BASE * (1 + self.REWARD_FOOD_HUNGER_SCALE * hunger_ratio**2)
            reward += base_reward
            self.expression = min(1.0, base_reward / self.EXPRESSION_SCALE)
            self.expression_time = time.time()
        
        # Starvation penalties
        hunger_ratio = self.hunger / self.max_hunger
        if hunger_ratio < 0.5:
            starvation_penalty = self.PENALTY_STARVATION_BASE * ((1 - hunger_ratio) / 0.5) ** 2
            reward += starvation_penalty
            if hunger_ratio < 0.2:
                self.expression = max(-1.0, starvation_penalty / self.EXPRESSION_SCALE)
                self.expression_time = time.time()
        
        # Movement penalties
        if action != 4:  # If not staying still
            # Calculate actual angle change (not target angle)
            angle_change = abs(self.angle - self.target_angle)
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            
            # Sharp turn penalty
            if angle_change > math.pi / 4:  # More than 45 degrees
                reward += self.PENALTY_SHARP_TURN * (angle_change / math.pi)
                self.expression = max(-1.0, self.PENALTY_SHARP_TURN / self.EXPRESSION_SCALE)
                self.expression_time = time.time()
            
            # Direction change penalty
            if action != self.prev_action:
                reward += self.PENALTY_DIRECTION_CHANGE
            
            # Smooth movement reward
            if angle_change < math.pi / 8:  # Less than 22.5 degrees
                reward += self.REWARD_SMOOTH_MOVEMENT
        
        # Wall collision penalty
        if wall_collision:
            reward += self.PENALTY_WALL
            # Add extra penalty if too close to walls
            wall_dist = min(new_head_x - margin, self.width - margin - new_head_x,
                          new_head_y - margin, self.height - margin - new_head_y)
            if wall_dist < self.head_size * 2:
                reward += self.PENALTY_WALL_STAY
            self.expression = max(-1.0, self.PENALTY_WALL / self.EXPRESSION_SCALE)
            self.expression_time = time.time()
        
        # Growth rewards when healthy
        if ate_plant and self.hunger > self.max_hunger * 0.5:
            reward += self.REWARD_GROWTH
        
        # Shrinking penalty
        if did_shrink:
            reward += self.PENALTY_SHRINK
            self.expression = max(-1.0, self.PENALTY_SHRINK / self.EXPRESSION_SCALE)
            self.expression_time = time.time()
        
        # Update previous action
        self.prev_action = action
        
        # Update episode reward
        self.episode_reward += reward
        
        # Check if episode is done and level up if score improved
        if not alive and self.episode_reward > self.level * 100:
            self.level += 1
        
        # Increment steps in level
        self.steps_in_level += 1
        
        # Check if level is complete
        if self.steps_in_level >= self.steps_for_level:  
            self.level += 1
            self.steps_in_level = 0
            self.steps_for_level = min(self.max_steps, self.steps_for_level + self.steps_increment)
        
        return self._get_state(), {
            'reward': reward,
            'alive': alive,
            'ate_plant': ate_plant,
            'wall_collision': wall_collision
        }
        
    def draw(self, surface=None):
        """Draw the game state"""
        if self.headless:
            return
            
        # Use provided surface or default game surface
        draw_surface = surface if surface is not None else self.game_surface
        
        # Fill background with dark color
        draw_surface.fill((20, 20, 20))
        
        # Draw walls as connected jagged lines
        def draw_wall_line(points):
            if len(points) > 1:
                pygame.draw.lines(draw_surface, self.wall_color, False, points, 2)
                # Add some darker shading on the inner edge
                darker_color = (max(0, self.wall_color[0] - 20),
                              max(0, self.wall_color[1] - 20),
                              max(0, self.wall_color[2] - 20))
                pygame.draw.lines(draw_surface, darker_color, False, points, 1)
        
        draw_wall_line(self.left_wall)
        draw_wall_line(self.right_wall)
        draw_wall_line(self.top_wall)
        draw_wall_line(self.bottom_wall)
        
        # Draw grid lines for visual reference
        grid_spacing = self.game_height // 10
        for i in range(0, self.game_width + grid_spacing, grid_spacing):
            pygame.draw.line(draw_surface, (40, 40, 40), (i, 0), (i, self.game_height))
        for i in range(0, self.game_height + grid_spacing, grid_spacing):
            pygame.draw.line(draw_surface, (40, 40, 40), (0, i), (self.game_width, i))
        
        # Draw plants
        for plant in self.plants:
            plant.draw(draw_surface)
        
        # Draw worm segments from tail to head
        for i in range(len(self.positions)-1, 0, -1):
            pos = self.positions[i]
            next_pos = self.positions[i-1]  # Next segment towards head
            
            # Calculate angle between segments
            dx = next_pos[0] - pos[0]
            dy = next_pos[1] - pos[1]
            angle = math.atan2(dy, dx)
            
            # Draw segment
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
        self._draw_segment(head_pos, head_angle, self.segment_width * 1.2, self.head_color, True)
        
        # Draw UI elements in top-right corner
        padding = 5
        meter_width = self.game_width // 6  # Smaller meters
        meter_height = self.game_height // 40  # Thinner meters
        meter_x = self.game_width - meter_width - padding * 2
        meter_y = padding * 2
        
        # Use smaller font for all text
        font = pygame.font.Font(None, self.game_height // 30)  # Smaller font
        
        # Draw hunger meter
        pygame.draw.rect(draw_surface, (100, 100, 100),
                       (meter_x, meter_y, meter_width, meter_height))
        
        # Draw hunger fill
        fill_width = int((meter_width - 2) * (self.hunger / self.max_hunger))
        fill_color = (
            int(255 * (1 - self.hunger/self.max_hunger)),
            int(255 * (self.hunger/self.max_hunger)),
            0
        )
        pygame.draw.rect(draw_surface, fill_color,
                       (meter_x+1, meter_y+1, fill_width, meter_height-2))
        
        # Draw "Hunger" label
        hunger_text = "Hunger"
        text_surface = font.render(hunger_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect()
        text_rect.right = meter_x - padding
        text_rect.centery = meter_y + meter_height//2
        draw_surface.blit(text_surface, text_rect)
        
        # Draw level progress bar
        progress_y = meter_y + meter_height + padding
        pygame.draw.rect(draw_surface, (100, 100, 100),
                       (meter_x, progress_y, meter_width, meter_height))
        
        # Draw progress fill
        progress = min(1.0, self.steps_in_level / self.steps_for_level)  
        fill_width = int((meter_width - 2) * progress)
        fill_color = (100, 200, 255)
        pygame.draw.rect(draw_surface, fill_color,
                       (meter_x+1, progress_y+1, fill_width, meter_height-2))
        
        # Draw "Level Progress" label
        progress_label = "Progress"  # Shortened for space
        text_surface = font.render(progress_label, True, (200, 200, 200))
        text_rect = text_surface.get_rect()
        text_rect.right = meter_x - padding
        text_rect.centery = progress_y + meter_height//2
        draw_surface.blit(text_surface, text_rect)
        
        # Draw stats in top-left corner
        stats_x = padding * 2
        stats_y = padding * 2
        line_height = self.game_height // 30
        
        # Draw stats with smaller font
        stats = [
            f"Level: {self.level}",
            f"Score: {int(self.episode_reward)}",
            f"Length: {self.num_segments}"
        ]
        
        for stat in stats:
            text_surface = font.render(stat, True, (200, 200, 200))
            text_rect = text_surface.get_rect()
            text_rect.topleft = (stats_x, stats_y)
            draw_surface.blit(text_surface, text_rect)
            stats_y += line_height
        
        # Draw steps counter below level progress bar
        steps_text = f"{self.steps_in_level}/{self.steps_for_level}"  
        text_surface = font.render(steps_text, True, (200, 200, 200))
        text_rect = text_surface.get_rect()
        text_rect.right = self.game_width - padding * 2
        text_rect.top = progress_y + meter_height + padding
        draw_surface.blit(text_surface, text_rect)
        
        # Draw game area border
        if not surface:
            # Clear screen
            screen = pygame.display.get_surface()
            screen.fill((20, 20, 20))
            
            # Draw game surface onto main screen with offset
            screen.blit(draw_surface, (self.game_x_offset, self.game_y_offset))
            
            # Draw border around game area
            pygame.draw.rect(screen, (100, 100, 100), 
                           (self.game_x_offset-2, self.game_y_offset-2, 
                            self.game_width+4, self.game_height+4), 2)
            
            pygame.display.flip()

    def _draw_segment(self, pos, angle, width, color, is_head=False):
        """Draw a single body segment"""
        x, y = pos
        
        # Draw main body circle
        pygame.draw.circle(self.game_surface, color, (int(x), int(y)), width)
        
        if is_head:
            # Rotate face 90 degrees to face movement direction
            face_angle = angle - math.pi/2  # Changed from + to - to rotate face forward
            
            # Calculate eye positions (rotate around head center)
            eye_offset = self.head_size * 0.3  # Distance from center
            eye_y_offset = self.head_size * 0.15  # Eyes moved up more (from 0.1)
            
            # Base eye positions (before rotation)
            base_left_x = -eye_offset
            base_left_y = -eye_y_offset
            base_right_x = eye_offset
            base_right_y = -eye_y_offset
            
            # Rotate eye positions based on face angle
            left_eye_x = x + (base_left_x * math.cos(face_angle) - base_left_y * math.sin(face_angle))
            left_eye_y = y + (base_left_x * math.sin(face_angle) + base_left_y * math.cos(face_angle))
            right_eye_x = x + (base_right_x * math.cos(face_angle) - base_right_y * math.sin(face_angle))
            right_eye_y = y + (base_right_x * math.sin(face_angle) + base_right_y * math.cos(face_angle))
            
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
            mouth_y_offset = self.head_size * 0.4  # Moved mouth down more (from 0.2)
            
            # Base mouth points (before rotation)
            base_left_x = -mouth_width/2
            base_left_y = mouth_y_offset
            base_right_x = mouth_width/2
            base_right_y = mouth_y_offset
            
            # Rotate mouth points based on face angle
            left_x = x + (base_left_x * math.cos(face_angle) - base_left_y * math.sin(face_angle))
            left_y = y + (base_left_x * math.sin(face_angle) + base_left_y * math.cos(face_angle))
            right_x = x + (base_right_x * math.cos(face_angle) - base_right_y * math.sin(face_angle))
            right_y = y + (base_right_x * math.sin(face_angle) + base_right_y * math.cos(face_angle))
            
            # Calculate control point for curved mouth
            curve_height = mouth_height * current_expression
            base_control_x = 0
            base_control_y = mouth_y_offset + curve_height
            
            control_x = x + (base_control_x * math.cos(face_angle) - base_control_y * math.sin(face_angle))
            control_y = y + (base_control_x * math.sin(face_angle) + base_control_y * math.cos(face_angle))
            
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
        
        # Use initial number of segments defined in __init__
        self.positions = [(self.x, self.y) for _ in range(self.num_segments)]
        
        # Reset colors
        self.body_colors = []
        for i in range(self.num_segments):
            green_val = int(150 - (100 * i / self.max_segments))
            self.body_colors.append((70, green_val + 30, 20))
        
        # Start with low hunger (30% of max) to encourage immediate food seeking
        self.hunger = self.max_hunger * 0.3
        
        # Clear and respawn plants
        self.plants = []
        self.target_plants = random.randint(self.min_plants, self.max_plants)
        initial_plants = random.randint(self.min_plants, self.target_plants)
        
        # Keep trying to spawn initial plants until we have enough
        attempts = 0
        while len(self.plants) < initial_plants and attempts < 100:
            self.spawn_plant()
            attempts += 1
            
        # Reset episode reward
        self.episode_reward = 0.0
        
        # Reset steps in level
        self.steps_in_level = 0
        
        return self._get_state()
        
    def update_hunger(self):
        """Update hunger and return whether worm is still alive"""
        old_hunger = self.hunger
        old_segments = self.num_segments
        self.hunger = max(0, self.hunger - self.current_hunger_rate)
        
        # Update shrink timer
        if self.shrink_timer > 0:
            self.shrink_timer -= 1
        
        # Calculate shrinking based on hunger level
        hunger_ratio = self.hunger / self.max_hunger
        if hunger_ratio < self.shrink_hunger_threshold and self.shrink_timer == 0:
            if self.num_segments > self.min_segments:
                # Show sad expression when shrinking
                self.expression = -1
                self.expression_time = time.time()
                # Remove last segment
                self.num_segments -= 1
                if len(self.positions) > self.num_segments:
                    self.positions.pop()
                    self.body_colors.pop()
                # Set cooldown
                self.shrink_timer = self.shrink_cooldown
                # Return shrink occurred
                return self.hunger > 0 and self.num_segments > 0, True
            elif old_hunger > 0 and self.hunger == 0:
                # Show very sad expression when at minimum size and starving
                self.expression = -1
                self.expression_time = time.time()
        
        # Die if no segments left or hunger is zero
        return self.hunger > 0 and self.num_segments > 0, False
        
class WormAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Increased memory size
        self.batch_size = 64  # Much smaller batch size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Slightly higher min exploration
        self.epsilon_decay = 0.9995  # Much slower decay
        self.learning_rate = 0.0005  # Slightly lower learning rate
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
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
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
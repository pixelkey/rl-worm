import pygame
import sys
import math
import random
import time
import numpy as np
from renderer import WormRenderer

class WormGame:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("AI Worm")
        self.clock = pygame.time.Clock()
        
        # Create renderer
        self.renderer = WormRenderer(width, height)
        
        # Game parameters
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        self.x = self.width // 2
        self.y = self.height // 2
        self.angle = 0
        self.speed = 2
        self.segment_size = 15
        self.num_segments = 5
        self.positions = [(self.x, self.y)]
        self.prev_action = 4  # Middle action (straight)
        
        # Food parameters
        self.food_size = 10
        self.food = []
        self.spawn_food()
        
        # Hunger system
        self.hunger = 0
        self.max_hunger = 1000
        self.hunger_rate = 1
        
        # Expression system
        self.expression = 0  # -1 to 1 (frown to smile)
        self.expression_time = time.time()
        self.expression_duration = 1.0  # How long expressions last
        
        return self._get_state()
    
    def spawn_food(self):
        """Spawn new food at random location"""
        if len(self.food) < 3:  # Maximum 3 food items
            margin = max(self.food_size, self.segment_size)
            x = random.randint(margin, self.width - margin)
            y = random.randint(margin, self.height - margin)
            self.food.append((x, y))
    
    def step(self, action):
        """Execute one step in the environment"""
        prev_x, prev_y = self.x, self.y
        
        # Update angle based on action (8 possible directions)
        target_angle = action * (2 * math.pi / 8)
        self.angle = target_angle
        
        # Move in the current direction
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # Boundary checking
        wall_collision = False
        if self.x < self.segment_size:
            self.x = self.segment_size
            wall_collision = True
        elif self.x > self.width - self.segment_size:
            self.x = self.width - self.segment_size
            wall_collision = True
        if self.y < self.segment_size:
            self.y = self.segment_size
            wall_collision = True
        elif self.y > self.height - self.segment_size:
            self.y = self.height - self.segment_size
            wall_collision = True
        
        # Update positions list
        self.positions.insert(0, (self.x, self.y))
        if len(self.positions) > self.num_segments:
            self.positions.pop()
        
        # Check for food collision
        ate_plant = False
        for food_pos in self.food[:]:
            if math.dist((self.x, self.y), food_pos) < self.segment_size + self.food_size:
                self.food.remove(food_pos)
                ate_plant = True
                self.spawn_food()
                self.num_segments += 1
                # Reduce hunger when eating
                self.hunger = max(0, self.hunger - 200)
        
        # Update hunger
        self.hunger = min(self.max_hunger, self.hunger + self.hunger_rate)
        
        # Update expression
        if time.time() - self.expression_time > self.expression_duration:
            # Gradually return to neutral expression
            self.expression *= 0.95
            if abs(self.expression) < 0.1:
                self.expression = 0
        
        # Calculate reward
        reward = 0
        
        # Small reward for moving (encourages exploration)
        movement_reward = 0.1
        distance_moved = math.sqrt((self.x - prev_x)**2 + (self.y - prev_y)**2)
        if distance_moved > 0:
            reward += movement_reward * (distance_moved / self.speed)
            self.expression = 0.1
            self.expression_time = time.time()
        
        # Reward for eating plants and growing
        if ate_plant:
            # Higher reward when hungrier (exponential scaling)
            hunger_ratio = 1 - (self.hunger / self.max_hunger)
            hunger_bonus = math.exp(hunger_ratio) - 1
            base_reward = 5.0 * (1 + 2 * hunger_bonus)
            reward += base_reward
            
            # Extra reward for growing
            growth_reward = 5.0
            reward += growth_reward
            
            # Set happy expression scaled by total reward and hunger
            total_reward = base_reward + growth_reward
            self.expression = min(1.0, (total_reward / 20.0) * (1 + hunger_ratio))
            self.expression_time = time.time()
        
        # Penalty for hitting walls
        if wall_collision:
            wall_penalty = -1.0
            reward += wall_penalty
            self.expression = wall_penalty / 2.0
            self.expression_time = time.time()
        
        # Penalty for sharp turns
        action_diff = abs(action - self.prev_action)
        if action_diff > 4:
            action_diff = 8 - action_diff
        if action_diff > 2:
            turn_penalty = -0.5 * (action_diff - 2)
            reward += turn_penalty
            self.expression = turn_penalty / 2.0
            self.expression_time = time.time()
        
        self.prev_action = action
        
        # Get new state
        state = self._get_state()
        done = False  # For now, game never ends
        
        return state, reward, done
    
    def _get_state(self):
        """Get the current state for the neural network"""
        # Normalize positions to [0, 1]
        norm_x = self.x / self.width
        norm_y = self.y / self.height
        
        # Get distances to nearest food
        food_distances = []
        food_angles = []
        if self.food:
            distances = [math.dist((self.x, self.y), food) for food in self.food]
            min_dist_idx = np.argmin(distances)
            nearest_food = self.food[min_dist_idx]
            
            # Normalize distance
            food_dist = distances[min_dist_idx] / math.sqrt(self.width**2 + self.height**2)
            food_distances.append(food_dist)
            
            # Calculate angle to food
            food_angle = math.atan2(nearest_food[1] - self.y, nearest_food[0] - self.x)
            # Normalize angle difference to [-1, 1]
            angle_diff = (food_angle - self.angle) / math.pi
            food_angles.append(angle_diff)
        else:
            food_distances.append(1.0)  # Max distance if no food
            food_angles.append(0.0)  # No angle if no food
        
        # Calculate distances to walls
        wall_dists = [
            self.x / self.width,  # Left wall
            (self.width - self.x) / self.width,  # Right wall
            self.y / self.height,  # Top wall
            (self.height - self.y) / self.height,  # Bottom wall
        ]
        
        # Normalize angle to [-1, 1]
        norm_angle = self.angle / math.pi
        
        # Normalize hunger
        norm_hunger = self.hunger / self.max_hunger
        
        # Previous action
        norm_prev_action = self.prev_action / 8.0
        
        # Combine all state components
        state = np.array([
            norm_x, norm_y,  # Position
            norm_angle,  # Current angle
            norm_prev_action,  # Previous action
            norm_hunger,  # Hunger level
            food_distances[0],  # Distance to nearest food
            food_angles[0],  # Angle to nearest food
            *wall_dists,  # Distances to walls
        ], dtype=np.float32)
        
        return state
    
    def render(self):
        """Render the game state"""
        # Prepare game state for renderer
        game_state = {
            'positions': self.positions,
            'food': self.food,
            'segment_size': self.segment_size,
            'food_size': self.food_size,
            'angle': self.angle,
            'expression': self.expression
        }
        
        # Get rendered surface from renderer
        game_surface = self.renderer.render_game(game_state)
        
        # Draw to screen
        self.screen.blit(game_surface, (0, 0))
        pygame.display.flip()
        
        # Control game speed
        self.clock.tick(60)
    
    def close(self):
        """Clean up resources"""
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = WormGame()
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.close()
            
            # Manual control for testing
            keys = pygame.key.get_pressed()
            action = 4  # Default to straight
            if keys[pygame.K_LEFT]:
                action = 6
            elif keys[pygame.K_RIGHT]:
                action = 2
            elif keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_DOWN]:
                action = 4
            
            game.step(action)
            game.render()
    except KeyboardInterrupt:
        game.close()
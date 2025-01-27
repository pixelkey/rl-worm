import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from analytics.metrics import WormAnalytics
import time
from datetime import datetime, timedelta
import sys
import os
from app import WormGame

# Set up environment variables for display
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["SDL_VIDEODRIVER"] = "x11"

# Initialize pygame (headless mode)
pygame.init()

# Training parameters
TRAINING_EPISODES = 1000
MIN_STEPS = 3000  # Start with more steps
MAX_STEPS = 8000  # Allow for longer episodes
STEPS_INCREMENT = 200  # Larger increment for better progression
PERFORMANCE_THRESHOLD = -50  # More lenient threshold
SAVE_INTERVAL = 10
PRINT_INTERVAL = 1

class ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.start_time = time.time()
        self.current = 0
        
    def update(self, current):
        self.current = current
        percentage = current / self.total
        filled = int(self.width * percentage)
        bar = '=' * filled + '-' * (self.width - filled)
        
        elapsed_time = time.time() - self.start_time
        if current > 0:
            estimated_total = elapsed_time * (self.total / current)
            remaining = estimated_total - elapsed_time
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = eta.strftime('%H:%M:%S')
        else:
            eta_str = "Calculating..."
            
        sys.stdout.write(f'\rProgress: [{bar}] {percentage:.1%} | Episode: {current}/{self.total} | ETA: {eta_str}')
        sys.stdout.flush()

class WormAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.batch_size = 512
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Try to load saved model
        self.load()

    def remember(self, state, action, reward, next_state, done):
        # Normalize reward to roughly [-1, 1] range
        # Base scale on wall hit (-200) and food reward (~100)
        normalized_reward = np.clip(reward / 200.0, -1.0, 1.0)
        
        # Ensure states are numpy arrays with correct dtype
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        self.memory.append((state, action, normalized_reward, next_state, done))
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 512),  
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )
    
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
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Pre-allocate and fill arrays
        states_array = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        next_states_array = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions_array = np.zeros(self.batch_size, dtype=np.int64)
        rewards_array = np.zeros(self.batch_size, dtype=np.float32)
        dones_array = np.zeros(self.batch_size, dtype=np.float32)
        
        # Fill arrays efficiently
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states_array[i] = state
            next_states_array[i] = next_state
            actions_array[i] = action
            rewards_array[i] = reward
            dones_array[i] = done
        
        # Convert all arrays to tensors at once
        states = torch.from_numpy(states_array).to(self.device)
        next_states = torch.from_numpy(next_states_array).to(self.device)
        actions = torch.from_numpy(actions_array).to(self.device)
        rewards = torch.from_numpy(rewards_array).to(self.device)
        dones = torch.from_numpy(dones_array).to(self.device)
        
        # Forward passes
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss and backward pass
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, episode):
        # Save only the model weights to prevent pickle security issues
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'saved')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'worm_model.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size
        }, model_path)
        print(f"Saved model state at episode {episode} to {model_path}")
        
    def load(self):
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'saved')
            model_path = os.path.join(model_dir, 'worm_model.pth')
            print(f"Attempting to load model from {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model file does not exist at {model_path}")
                print("No saved model found, starting fresh")
                return
                
            checkpoint = torch.load(model_path, map_location=self.device)
            if checkpoint['state_size'] == self.state_size:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Successfully loaded model with epsilon: {self.epsilon:.4f}")
            else:
                print(f"Model has wrong state size: expected {self.state_size}, got {checkpoint['state_size']}")
                print("Starting fresh with new model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("No saved model found, starting fresh")
            
        # Always sync target model with main model after loading
        self.target_model.load_state_dict(self.model.state_dict())
    
def fast_training():
    print("\nStarting fast training mode...")
    print("Press Ctrl+C at any time to stop training and save progress\n")
    
    progress = ProgressBar(TRAINING_EPISODES)
    start_time = time.time()
    
    # Initialize agent and analytics
    STATE_SIZE = 14  # [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, plant_dist, plant_dx, plant_dy, wall_left, wall_right, wall_top, wall_bottom, hunger]
    ACTION_SIZE = 9
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    analytics = WormAnalytics()
    
    steps_per_episode = MIN_STEPS
    last_time = time.time()
    
    try:
        for episode in range(TRAINING_EPISODES):
            # Initialize game state
            game = WormGame(headless=True)
            state = game.reset()
            total_reward = 0
            plants_eaten = 0
            steps_survived = 0
            episode_positions = set()
            wall_collisions = 0
            wall_stays = 0
            danger_zone_count = 0
            direction_changes = 0
            sharp_turns = 0
            smooth_movements = 0
            starvation_count = 0
            shrink_count = 0
            food_rewards = 0
            growth_rewards = 0
            exploration_rewards = 0
            
            # Track performance
            training_time = 0
            game_time = 0
            
            for step in range(steps_per_episode):
                step_start = time.time()
                
                # Get action and update game
                action = agent.act(state)
                game_start = time.time()
                next_state, reward, done, info = game.step(action)
                game_time += time.time() - game_start
                
                # Track metrics
                if info['ate_plant']:
                    plants_eaten += 1
                
                reward_source = game.last_reward_source
                
                # Food rewards
                if 'Food' in reward_source:
                    food_rewards += 1
                    if ' + Growth' in reward_source:
                        growth_rewards += 1
                
                # Movement rewards (mutually exclusive)
                if 'Sharp Turn' in reward_source:
                    sharp_turns += 1
                elif 'Direction Change' in reward_source:
                    direction_changes += 1
                elif 'Smooth Movement' in reward_source:
                    smooth_movements += 1
                
                # Exploration rewards
                if 'Exploration' in reward_source:
                    exploration_rewards += 1
                
                # Wall penalties
                if 'Wall Collision' in reward_source:
                    wall_collisions += 1
                if 'Wall Stay' in reward_source:
                    wall_stays += 1
                if 'Danger Zone' in reward_source:
                    danger_zone_count += 1
                
                # Health penalties (mutually exclusive)
                if 'Starvation' in reward_source:
                    starvation_count += 1
                elif 'Shrinking' in reward_source:
                    shrink_count += 1
                
                # Track visited positions
                head_pos = game.positions[0]
                grid_size = game.game_width / 20
                grid_pos = (int(head_pos[0]/grid_size), int(head_pos[1]/grid_size))
                episode_positions.add(grid_pos)
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                
                train_start = time.time()
                if len(agent.memory) >= agent.batch_size and step % 4 == 0:
                    loss = agent.train()
                training_time += time.time() - train_start
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                steps_survived += 1
                
                # Update target network periodically
                if step % 1000 == 0:
                    agent.update_target_model()
                
                # Break if done
                if done:
                    break
            
            # Calculate metrics
            exploration_ratio = len(episode_positions) / (steps_survived * 0.25)  
            
            # Update analytics
            metrics = {
                'avg_reward': total_reward,
                'wall_collisions': wall_collisions,
                'wall_stays': wall_stays,
                'danger_zone_count': danger_zone_count,
                'exploration_ratio': exploration_ratio,
                'movement_smoothness': steps_survived / steps_per_episode,
                'epsilon': agent.epsilon
            }
            
            analytics.update_metrics(episode, metrics)
            
            # Save model periodically
            if episode % SAVE_INTERVAL == 0:
                agent.save(episode)
            
            # Update progress bar
            progress.update(episode + 1)
            
            # Print metrics periodically
            if episode % PRINT_INTERVAL == 0:
                current_time = time.time()
                episode_time = current_time - last_time
                print(f"\nEpisode {episode}/{TRAINING_EPISODES}")
                print(f"Steps: {steps_survived}/{steps_per_episode}")
                print(f"Reward: {total_reward:.2f}")
                print(f"Plants: {plants_eaten} (Food: {food_rewards}, Growth: {growth_rewards})")
                print(f"Wall: {wall_collisions} hits, {wall_stays} stays, {danger_zone_count} nears")
                print(f"Movement: {smooth_movements} (Sharp: {sharp_turns}, Dir: {direction_changes})")
                print(f"Health: {shrink_count} shrinks, {starvation_count} starves")
                print(f"Explore: {exploration_ratio:.2f} ({exploration_rewards} rewards)")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Episode Time: {episode_time:.2f}s")
                print(f"Game Time: {game_time:.2f}s")
                print(f"Training Time: {training_time:.2f}s")
                last_time = current_time
            
            # Increase steps if performance improves
            if total_reward > PERFORMANCE_THRESHOLD and steps_per_episode < MAX_STEPS:
                steps_per_episode += STEPS_INCREMENT
                print(f"Steps increased to {steps_per_episode}")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving progress...")
        agent.save(episode)
        analytics.generate_report(episode)
        
    print("\nTraining finished!")
    agent.save(TRAINING_EPISODES)
    analytics.generate_report(TRAINING_EPISODES)

if __name__ == "__main__":
    try:
        fast_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        analytics.generate_report(episode)
        print("Progress saved. Exiting...")
        print("\nTraining interrupted by user. Saving progress...")
        analytics.generate_report(episode)
        print("Progress saved. Exiting...")

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
MIN_STEPS = 2000  # Start with 2000 steps
MAX_STEPS = 5000  # Maximum steps per episode
STEPS_INCREMENT = 100  # How many steps to add when performance improves
PERFORMANCE_THRESHOLD = 200  # Only increase steps for positive average reward
SAVE_INTERVAL = 10
PRINT_INTERVAL = 1

# Add reward window for smoother step increases
REWARD_WINDOW_SIZE = 5  # Average over last 5 episodes
reward_history = []

def should_increase_steps(reward, steps_per_episode):
    """Determine if we should increase steps based on performance"""
    reward_history.append(reward)
    if len(reward_history) > REWARD_WINDOW_SIZE:
        reward_history.pop(0)
        avg_reward = sum(reward_history) / len(reward_history)
        return avg_reward > PERFORMANCE_THRESHOLD and steps_per_episode < MAX_STEPS
    return False

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
        self.batch_size = 256  # Smaller batch size for faster training
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.997  # Slower decay
        self.learning_rate = 0.001  # Slightly higher learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', 
                                                            factor=0.5, patience=5,
                                                            verbose=True)
        
        # Try to load saved model
        self.load()
    
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
    
    def remember(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays with correct dtype
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
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
    STATE_SIZE = 28  # Updated to match app.py: pos(2) + vel(2) + acc(2) + angles(2) + energy(1) + walls(4) + plant_info(15)
    ACTION_SIZE = 9
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    analytics = WormAnalytics()
    
    steps_per_episode = MIN_STEPS
    last_time = time.time()
    episode_rewards = []  # Track episode rewards for logging
    
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
            
            # Track performance
            training_time = 0
            game_time = 0
            
            for step in range(steps_per_episode):
                step_start = time.time()
                
                # Get action and update game
                action = agent.act(state)
                next_state, reward, done, info = game.step(action)
                
                # Track metrics
                if info['ate_plant']:
                    plants_eaten += 1
                if info['wall_collision']:
                    wall_collisions += 1
                
                # Track visited positions for exploration metric
                head_pos = game.positions[0]
                grid_size = game.game_width / 20
                grid_pos = (int(head_pos[0] / grid_size), int(head_pos[1] / grid_size))
                episode_positions.add(grid_pos)
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                loss = agent.train()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
                
                # Update timing metrics
                step_end = time.time()
                if loss is not None:
                    training_time += step_end - step_start
                game_time += step_end - step_start
            
            # Calculate exploration ratio
            total_grids = (game.game_width / grid_size) * (game.game_height / grid_size)
            exploration_ratio = len(episode_positions) / total_grids
            
            # Track episode statistics
            episode_rewards.append(total_reward)
            avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
            
            # Only increase steps if performance is good
            if should_increase_steps(total_reward, steps_per_episode):
                steps_per_episode = min(steps_per_episode + STEPS_INCREMENT, MAX_STEPS)
                print(f"Steps increased to {steps_per_episode}")
            
            # Print detailed episode stats
            print(f"\nEpisode {episode}/{TRAINING_EPISODES}")
            print(f"Steps: {step+1}/{steps_per_episode}")
            print(f"Reward: {total_reward:.2f}")
            print(f"Avg Reward (10 ep): {avg_reward:.2f}")
            print(f"Plants: {plants_eaten}")
            print(f"Wall Hits: {wall_collisions}")
            print(f"Explore: {exploration_ratio:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Episode Time: {game_time:.2f}s")
            print(f"Training Time: {training_time:.2f}s")
            
            # Update progress bar
            progress.update(episode + 1)
            
            # Save model periodically
            if (episode + 1) % SAVE_INTERVAL == 0:
                agent.save(episode)
                
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

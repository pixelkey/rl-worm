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
from app import WormGame
import torch.amp

# Initialize pygame (headless mode)
pygame.init()

# Training parameters
TRAINING_EPISODES = 1000
MIN_STEPS = 2000  # Start with 2000 steps
MAX_STEPS = 5000  # Maximum steps per episode
STEPS_INCREMENT = 100  # How many steps to add when performance improves
PERFORMANCE_THRESHOLD = -100  # Reward threshold to increase steps
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
        self.memory = deque(maxlen=100000)  # Increased from 50000
        self.batch_size = 4096  # Increased from 2048 for faster learning
        self.training_steps_per_update = 1  # Single efficient update
        self.training_frequency = 1  # Train every step
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mixed precision scaler
        self.scaler = torch.amp.GradScaler()
        
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9, verbose=True)
        
        # Try to load saved model
        self.load()
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 512),  # Increased from 256
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for regularization
            nn.Linear(512, 512),  # Increased from 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),  # Increased from 128
            nn.ReLU(),
            nn.Linear(256, self.action_size)
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
            
        # Sample batch all at once
        indices = random.sample(range(len(self.memory)), self.batch_size)
        batch = [self.memory[i] for i in indices]
        
        # Convert to tensors efficiently
        states = torch.tensor([s[0] for s in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([s[1] for s in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([s[3] for s in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([s[4] for s in batch], dtype=torch.float32, device=self.device)
        
        # Enable mixed precision training
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            current_q = self.model(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
            
            loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimizer steps with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, episode):
        # Save only the model weights to prevent pickle security issues
        model_path = f'models/saved/worm_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size
        }, model_path)
        print(f"Saved model state at episode {episode}")
        
    def load(self):
        try:
            model_path = 'models/saved/worm_model.pth'
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if checkpoint['state_size'] == self.state_size:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Loaded training state. Epsilon: {self.epsilon:.4f}")
            else:
                print("Old model architecture detected, starting fresh")
        except:
            print("No saved model found, starting fresh")
            
        # Always sync target model with main model after loading
        self.target_model.load_state_dict(self.model.state_dict())
    
def fast_training():
    print("\nStarting fast training mode...")
    print("Press Ctrl+C at any time to stop training and save progress\n")
    
    progress = ProgressBar(TRAINING_EPISODES)
    start_time = time.time()
    
    # Initialize agent and analytics
    STATE_SIZE = 14  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), walls (4), hunger (1)
    ACTION_SIZE = 9  # 8 directions + no movement
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
            
            # Track performance
            training_time = 0
            game_time = 0
            
            for step in range(steps_per_episode):
                step_start = time.time()
                
                # Get action and update game
                action = agent.act(state)
                next_state, info = game.step(action)
                
                # Get reward from game
                reward = info.get('reward', 0)
                done = not info['alive']
                
                game_time += time.time() - step_start
                
                # Track metrics
                if info['ate_plant']:
                    plants_eaten += 1
                if info['wall_collision']:
                    wall_collisions += 1
                
                # Track visited positions
                head_pos = game.positions[0]
                grid_size = game.game_width / 20
                grid_pos = (int(head_pos[0]/grid_size), int(head_pos[1]/grid_size))
                episode_positions.add(grid_pos)
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                
                train_start = time.time()
                if step % agent.training_frequency == 0 and len(agent.memory) >= agent.batch_size:
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
            exploration_ratio = len(episode_positions) / (steps_survived * 0.25)  # Assuming we want to explore 25% of possible positions
            
            # Update analytics
            metrics = {
                'avg_reward': total_reward,
                'wall_collisions': wall_collisions,
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
                print(f"Plants: {plants_eaten}")
                print(f"Explore: {exploration_ratio:.2f}")
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

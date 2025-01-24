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

# Initialize pygame (headless mode)
pygame.init()

# Training parameters
TRAINING_EPISODES = 1000
STEPS_PER_EPISODE = 2000
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
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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

def fast_training():
    print("\nStarting fast training mode...")
    print("Press Ctrl+C at any time to stop training and save progress\n")
    
    progress = ProgressBar(TRAINING_EPISODES)
    start_time = time.time()
    
    # Initialize agent and analytics
    STATE_SIZE = 12  # position (2), velocity (2), walls (4), plant info (4)
    ACTION_SIZE = 9  # 8 directions + no movement
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    analytics = WormAnalytics()
    
    # Training metrics
    training_metrics = {
        'episode': [],
        'avg_reward': [],
        'plants_eaten': [],
        'survival_time': [],
        'exploration_ratio': [],
        'epsilon': []
    }
    
    recent_rewards = []  # Keep track of last 50 episodes
    
    for episode in range(TRAINING_EPISODES):
        # Initialize game state
        game = WormGame()
        state = game.reset()
        total_reward = 0
        plants_eaten = 0
        steps_survived = 0
        episode_positions = set()
        
        for step in range(STEPS_PER_EPISODE):
            # Get action and update game
            action = agent.act(state)
            next_state, info = game.step(action)
            
            # Track metrics
            if info['ate_plant']:
                plants_eaten += 1
            
            # Track visited positions
            head_pos = (int(game.x/50), int(game.y/50))
            episode_positions.add(head_pos)
            
            # Get reward from game
            reward = info.get('reward', 0)
            done = not info['alive']
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train()
            
            if done:
                break
                
            state = next_state
            total_reward += reward
            steps_survived += 1
            
            # Update target network periodically
            if step % 1000 == 0:
                agent.update_target_model()
        
        # Update recent rewards
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)
        
        # Update analytics
        exploration_ratio = len(episode_positions) / ((game.width//50) * (game.height//50))
        training_metrics['episode'].append(episode)
        training_metrics['avg_reward'].append(avg_reward)
        training_metrics['plants_eaten'].append(plants_eaten)
        training_metrics['survival_time'].append(steps_survived)
        training_metrics['exploration_ratio'].append(exploration_ratio)
        training_metrics['epsilon'].append(agent.epsilon)
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            agent.save(episode)
            analytics.plot_training_metrics(training_metrics)
        
        # Update progress bar and print metrics
        progress.update(episode + 1)
        
        if episode % PRINT_INTERVAL == 0:
            print(f"\nEpisode {episode}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Average Reward (last 50): {avg_reward:.2f}")
            print(f"Plants Eaten: {plants_eaten}")
            print(f"Steps Survived: {steps_survived}")
            print(f"Exploration Ratio: {exploration_ratio:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Loss: {loss if loss is not None else 'N/A'}")
            
    print("\nTraining finished!")
    agent.save(TRAINING_EPISODES)
    analytics.plot_training_metrics(training_metrics)

if __name__ == "__main__":
    try:
        fast_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        analytics.plot_training_metrics(training_metrics)
        print("Progress saved. Exiting...")

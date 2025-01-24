import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import json
import math

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Larger network for better GPU utilization
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Initialize weights for better training
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state).to(self.device),
            torch.LongTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(done).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

class WormAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01  # Lower minimum epsilon
        self.epsilon_decay = 0.995  # Slower decay
        self.learning_rate = learning_rate
        self.memory = ReplayBuffer(50000)  # Larger replay buffer
        self.batch_size = 512  # Larger batch size for better GPU utilization
        self.update_target_every = 1000  # Update target network more frequently
        self.steps = 0
        
        # Create models directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Initialize models and move to GPU
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Use Adam with better parameters
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.epsilons = []
        
        # Try to load previous training state
        self.load_state()

    def get_state(self, worm_positions, velocity, screen_size):
        head_pos = worm_positions[-1]
        distances_to_walls = [
            head_pos[0],  # distance to left wall
            screen_size[0] - head_pos[0],  # distance to right wall
            head_pos[1],  # distance to top wall
            screen_size[1] - head_pos[1]  # distance to bottom wall
        ]
        
        # Normalize all values
        normalized_pos = [head_pos[0]/screen_size[0], head_pos[1]/screen_size[1]]
        normalized_vel = [velocity[0]/5.0, velocity[1]/5.0]  # assuming max speed is 5
        normalized_distances = [d/max(screen_size) for d in distances_to_walls]
        
        state = normalized_pos + normalized_vel + normalized_distances
        return np.array(state, dtype=np.float32)

    def get_action_movement(self, action):
        """Convert action index to movement vector"""
        if action == 8:  # No movement
            return [0, 0]
        
        angle = action * (2 * math.pi / 8)
        return [math.cos(angle) * 2, math.sin(angle) * 2]

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        self.steps += 1
        
        # Sample multiple batches for better GPU utilization
        n_batches = 4
        total_loss = 0
        
        for _ in range(n_batches):
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Use torch.amp.autocast instead of deprecated torch.cuda.amp.autocast
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Compute current Q values
                current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
                
                # Compute next Q values
                with torch.no_grad():
                    next_q_values = self.target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Compute loss and optimize
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        # Update target network
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss / n_batches

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_state(self, episode, is_best=False):
        """Save the model and training state"""
        if is_best:
            state_path = 'models/saved/worm_state_best.pt'
            metrics_path = 'models/saved/training_metrics_best.json'
        else:
            state_path = 'models/saved/worm_state.pt'
            metrics_path = 'models/saved/training_metrics.json'
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode
        }, state_path)
        
        # Save metrics
        metrics = {
            'losses': self.losses[-1000:],  # Save last 1000 entries
            'rewards': self.rewards[-1000:],
            'epsilons': self.epsilons[-1000:]
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
    def load_state(self, use_best=False):
        """Load the model and training state if available"""
        if use_best:
            state_path = 'models/saved/worm_state_best.pt'
            metrics_path = 'models/saved/training_metrics_best.json'
        else:
            state_path = 'models/saved/worm_state.pt'
            metrics_path = 'models/saved/training_metrics.json'
        
        if os.path.exists(state_path):
            try:
                checkpoint = torch.load(state_path, weights_only=True)  # Add weights_only=True
                
                # Handle old model architecture gracefully
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.epsilon = checkpoint['epsilon']
                except Exception as e:
                    print("Old model architecture detected, starting fresh")
                    return False
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        self.losses = metrics.get('losses', [])
                        self.rewards = metrics.get('rewards', [])
                        self.epsilons = metrics.get('epsilons', [])
                        
                print(f"Loaded {'best ' if use_best else ''}training state. Epsilon: {self.epsilon:.4f}")
                return True
            except Exception as e:
                print(f"Error loading model state: {e}")
                print("Starting fresh.")
                return False
        else:
            print(f"No {'best ' if use_best else ''}training state found. Starting fresh.")
            return False

import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
                
                # Sync target model with main model after loading
                self.target_model.load_state_dict(self.model.state_dict())
            else:
                print(f"Model has wrong state size: expected {self.state_size}, got {checkpoint['state_size']}")
                print("Starting fresh with new model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("No saved model found, starting fresh")

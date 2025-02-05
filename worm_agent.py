import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import MODEL_DIR, MODEL_PATH

class WormBrain(nn.Module):
    def __init__(self, state_size, action_size):
        super(WormBrain, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Movement action head
        self.action_head = nn.Linear(256, action_size)
        
        # Plant targeting head (3 plants)
        self.target_head = nn.Linear(256, 3)
        
    def forward(self, x):
        shared_features = self.shared(x)
        return self.action_head(shared_features), self.target_head(shared_features)

class WormAgent:
    def __init__(self, state_size, action_size):
        """Initialize an agent
        
        Args:
            state_size (int): Size of state space
            action_size (int): Size of action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        
        self.model = WormBrain(state_size, action_size).to(self.device)
        self.target_model = WormBrain(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
    def remember(self, state, action, target, reward, next_state, done):
        """Add experience to memory"""
        # Normalize reward to roughly [-1, 1] range
        # Base scale on wall hit (-200) and food reward (~100)
        normalized_reward = np.clip(reward / 200.0, -1.0, 1.0)
        
        # Convert state and next_state to float32 before storing
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        self.memory.append((state, action, target, normalized_reward, next_state, done))
    
    def act(self, state):
        """Choose an action and target plant based on state"""
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            target_plant = random.randrange(3)
            return action, target_plant
            
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, target_values = self.model(state)
            action = torch.argmax(action_values).item()
            target_plant = torch.argmax(target_values).item()
            
        return action, target_plant

    def train(self):
        """Train the agent using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        states_array = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        next_states_array = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions_array = np.zeros(self.batch_size, dtype=np.int64)
        targets_array = np.zeros(self.batch_size, dtype=np.int64)
        rewards_array = np.zeros(self.batch_size, dtype=np.float32)
        dones_array = np.zeros(self.batch_size, dtype=np.bool_)
        
        for i, (state, action, target, reward, next_state, done) in enumerate(minibatch):
            states_array[i] = state
            next_states_array[i] = next_state
            actions_array[i] = action
            targets_array[i] = target
            rewards_array[i] = reward
            dones_array[i] = done
            
        states = torch.FloatTensor(states_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        targets = torch.LongTensor(targets_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        dones = torch.BoolTensor(dones_array).to(self.device)
        
        # Get current Q values for both heads
        current_action_q, current_target_q = self.model(states)
        current_action_q = current_action_q.gather(1, actions.unsqueeze(1))
        current_target_q = current_target_q.gather(1, targets.unsqueeze(1))
        
        # Get next Q values
        next_action_q, next_target_q = self.target_model(next_states)
        max_next_action_q = next_action_q.max(1)[0]
        max_next_target_q = next_target_q.max(1)[0]
        
        # Compute target Q values
        target_action_q = rewards + (self.gamma * max_next_action_q * ~dones)
        target_target_q = rewards + (self.gamma * max_next_target_q * ~dones)
        
        # Compute loss for both heads
        action_loss = F.mse_loss(current_action_q.squeeze(), target_action_q)
        target_loss = F.mse_loss(current_target_q.squeeze(), target_target_q)
        total_loss = action_loss + target_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss.item()
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, episode):
        # Save only the model weights to prevent pickle security issues
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size
        }, MODEL_PATH)
        print(f"Saved model state at episode {episode} to {MODEL_PATH}")

    def load(self):
        try:
            print(f"Attempting to load model from {MODEL_PATH}")
            
            if not os.path.exists(MODEL_PATH):
                print(f"Model file does not exist at {MODEL_PATH}")
                print("No saved model found, starting fresh")
                return
                
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
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

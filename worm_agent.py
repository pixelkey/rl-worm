import os
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import MODEL_DIR, MODEL_PATH, MEMORY_SIZE, BATCH_SIZE

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
        """
        Initialize an agent.
        Args:
            state_size (int): Size of state space.
            action_size (int): Size of action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        # Updated memory size from config
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # Updated batch size from config
        self.batch_size = BATCH_SIZE  
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
        """
        Add experience to memory.
        Adjusted reward normalization: we now divide by 100 and clip to [-3, 3] 
        so that severe penalties (like wall collisions) remain distinct.
        """
        normalized_reward = np.clip(reward / 100.0, -3.0, 3.0)
        # Ensure states are stored as float32
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, target, normalized_reward, next_state, done))
    
    def act(self, state, epsilon):
        """
        Choose an action and target plant based on state, using the provided epsilon.
        """
        if random.random() <= epsilon:
            action = random.randrange(self.action_size)
            target_plant = random.randrange(3)
            return action, target_plant
            
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values, target_values = self.model(state)
            action = torch.argmax(action_values).item()
            target_plant = torch.argmax(target_values).item()
        return action, target_plant

    def train(self):
        """
        Train the agent using experience replay.
        """
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
        
        # Current Q values for both heads
        current_action_q, current_target_q = self.model(states)
        current_action_q = current_action_q.gather(1, actions.unsqueeze(1))
        current_target_q = current_target_q.gather(1, targets.unsqueeze(1))
        
        # Next Q values from the target network
        next_action_q, next_target_q = self.target_model(next_states)
        max_next_action_q = next_action_q.max(1)[0]
        max_next_target_q = next_target_q.max(1)[0]
        
        # Compute target Q values
        target_action_q = rewards + (self.gamma * max_next_action_q * ~dones)
        target_target_q = rewards + (self.gamma * max_next_target_q * ~dones)
        
        # Calculate loss for both heads
        action_loss = F.mse_loss(current_action_q.squeeze(), target_action_q)
        target_loss = F.mse_loss(current_target_q.squeeze(), target_target_q)
        total_loss = action_loss + target_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Removed internal epsilon decay for consistency; epsilon is now controlled externally.
        return total_loss.item()
    
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, episode):
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
                print(f"Model file does not exist at {MODEL_PATH}. Starting fresh.")
                return
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            if checkpoint['state_size'] == self.state_size:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                print(f"Successfully loaded model with epsilon: {self.epsilon:.4f}")
                self.target_model.load_state_dict(self.model.state_dict())
            else:
                print(f"Model state size mismatch: expected {self.state_size}, got {checkpoint['state_size']}. Starting fresh.")
        except Exception as e:
            print(f"Error loading model: {str(e)}. Starting fresh.")

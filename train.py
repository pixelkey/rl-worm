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
from worm_agent import WormAgent

# Set up environment variables for display
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["SDL_VIDEODRIVER"] = "x11"

# Initialize pygame (headless mode)
pygame.init()

# Training parameters
TRAINING_EPISODES = 1000
MIN_STEPS = 1000  # Starting number of steps
MAX_STEPS = 6000  # Maximum steps allowed
STEPS_INCREMENT = 50  # Changed from 200 to 50 steps per episode
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

def fast_training():
    print("\nStarting fast training mode...")
    print("Press Ctrl+C at any time to stop training and save progress\n")
    
    progress = ProgressBar(TRAINING_EPISODES)
    start_time = time.time()
    
    # Initialize agent and analytics
    STATE_SIZE = 15  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), plant_value (1), walls (4), hunger (1)
    ACTION_SIZE = 9
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    analytics = WormAnalytics()
    
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
            
            steps_per_episode = MIN_STEPS + (episode * STEPS_INCREMENT)  # Add increment each episode
            if steps_per_episode > MAX_STEPS:
                steps_per_episode = MAX_STEPS
            
            for step in range(steps_per_episode):
                step_start = time.time()
                
                # Get action from agent
                action, target_plant = agent.act(state)
                
                # Execute action in environment
                next_state, reward, done, info = game.step((action, target_plant))
                
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
                agent.remember(state, action, target_plant, reward, next_state, done)
                
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

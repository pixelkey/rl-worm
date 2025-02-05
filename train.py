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
import json
from app import WormGame
from worm_agent import WormAgent
from config import (
    STATE_SIZE, ACTION_SIZE,
    TRAINING_EPISODES, STARTING_STEPS, MAX_STEPS, STEPS_INCREMENT,
    PERFORMANCE_THRESHOLD, SAVE_INTERVAL, PRINT_INTERVAL,
    MODEL_DIR, MODEL_PATH, CHECKPOINT_PATH,
    EPSILON_START, EPSILON_FINAL, EPSILON_DECAY
)

# Set up environment variables for display
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["SDL_VIDEODRIVER"] = "x11"

# Initialize pygame (headless mode)
pygame.init()

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
    
    # Initialize agent and analytics
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    
    # Initialize or load step count from checkpoint
    checkpoint_path = CHECKPOINT_PATH
    model_path = MODEL_PATH
    start_episode = 0
    if os.path.exists(model_path):
        print(f"Found model at: {model_path}")
        try:
            # Get a sample of initial weights for verification
            initial_weights = next(agent.model.parameters()).clone().data[0][0].item()
            print(f"Initial model weights (sample): {initial_weights:.6f}")
            
            # Use agent's built-in load method
            agent.load()
            
            # Verify weights changed after loading
            loaded_weights = next(agent.model.parameters()).clone().data[0][0].item()
            print(f"Loaded model weights (sample): {loaded_weights:.6f}")
            
            if abs(initial_weights - loaded_weights) < 1e-6:
                print("WARNING: Model weights appear unchanged after loading!")
            else:
                print("Successfully loaded model weights (verified different from initial weights)")
                
            # Quick prediction test with both outputs
            test_input = torch.zeros((1, STATE_SIZE)).to(agent.device)
            action_output, target_output = agent.model(test_input)
            initial_action = action_output.argmax().item()
            initial_target = target_output.argmax().item()
            print(f"Model prediction test - action: {initial_action}, target: {initial_target}")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            
        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint at: {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                global STARTING_STEPS
                STARTING_STEPS = checkpoint_data.get('steps', STARTING_STEPS)
                start_episode = checkpoint_data.get('episode', 0)
                epsilon = checkpoint_data.get('epsilon', EPSILON_START)  # Load epsilon from checkpoint
                print(f"Resuming from episode {start_episode} with step count: {STARTING_STEPS} and epsilon: {epsilon:.3f}")
        else:
            print(f"No checkpoint found at: {checkpoint_path}")
    else:
        print(f"No model found at: {model_path}")

    # Initialize progress bar with total episodes, not just remaining
    progress = ProgressBar(TRAINING_EPISODES)
    # Update progress bar to current episode if resuming
    if start_episode > 0:
        progress.update(start_episode)
    start_time = time.time()
    
    # Initialize analytics
    analytics = WormAnalytics()
    
    last_time = time.time()
    
    try:
        current_steps = STARTING_STEPS
        for episode in range(start_episode, TRAINING_EPISODES):
            epsilon = max(EPSILON_FINAL, EPSILON_START * (EPSILON_DECAY ** episode))
            print(f"Episode {episode+1}/{TRAINING_EPISODES}, Epsilon: {epsilon:.3f}")
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
            
            # Use current_steps instead of calculating based on episode
            steps_per_episode = min(current_steps, MAX_STEPS)
            
            for step in range(steps_per_episode):
                step_start = time.time()
                
                # Get action from agent
                action, target_plant = agent.act(state, epsilon)
                
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
            
            # Only increment steps if worm survived the full episode
            if steps_survived >= steps_per_episode:
                current_steps += STEPS_INCREMENT
                print(f"\nWorm survived full episode! Increasing steps to: {current_steps}")
            
            # Check if the worm completes the max number of steps
            if steps_survived >= MAX_STEPS:
                print(f"\nWorm completed max steps in episode {episode}. Saving model...")
                agent.save(episode)
                analytics.generate_report(episode)
                break
            
            # Calculate metrics
            exploration_ratio = len(episode_positions) / (steps_survived * 0.25)  
            
            # Track death statistics
            death_penalty_total = 0
            deaths_by_starvation = 0
            deaths_by_wall = 0
            
            if game.death_cause == "starvation":
                deaths_by_starvation = 1
                death_penalty_total += game.PENALTY_DEATH
            elif game.death_cause == "wall_collision":
                deaths_by_wall = 1
                death_penalty_total += game.PENALTY_DEATH
            
            # Update analytics
            metrics = {
                'avg_reward': total_reward,
                'wall_collisions': wall_collisions,
                'wall_stays': wall_stays,
                'danger_zone_count': danger_zone_count,
                'exploration_ratio': exploration_ratio,
                'movement_smoothness': steps_survived / steps_per_episode,
                'epsilon': epsilon,
                'deaths_by_starvation': deaths_by_starvation,
                'deaths_by_wall': deaths_by_wall,
                'death_penalty_total': death_penalty_total
            }
            
            analytics.update_metrics(episode, metrics)
            
            # Save model periodically
            if episode % SAVE_INTERVAL == 0:
                # Use agent's built-in save method
                agent.save(episode)
                # Save both steps and episode to checkpoint
                checkpoint_data = {
                    'episode': episode + 1,
                    'steps': current_steps,  # Save current_steps instead of steps_per_episode
                    'epsilon': epsilon  # Save current epsilon value
                }
                os.makedirs(MODEL_DIR, exist_ok=True)
                with open(CHECKPOINT_PATH, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"Saved checkpoint at episode {episode}")
            
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
                if game.death_cause:
                    print(f"Death Cause: {game.death_cause}")
                print(f"Explore: {exploration_ratio:.2f} ({exploration_rewards} rewards)")
                print(f"Epsilon: {epsilon:.3f}")
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

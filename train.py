import pygame
import random
import math
import numpy as np
from models.dqn import WormAgent
from analytics.metrics import WormAnalytics
import time
from datetime import datetime, timedelta
import sys

# Initialize pygame (headless mode)
pygame.init()
width, height = 800, 600

# Training parameters
TRAINING_EPISODES = 1000
STEPS_PER_EPISODE = 2000
SAVE_INTERVAL = 10
PRINT_INTERVAL = 1  # Update every episode

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
    
    best_reward = float('-inf')
    episode_rewards = []
    recent_rewards = []  # Keep track of last 50 episodes
    
    # Initialize agent and analytics
    STATE_SIZE = 8
    ACTION_SIZE = 9
    agent = WormAgent(STATE_SIZE, ACTION_SIZE)
    analytics = WormAnalytics()
    
    # Track visited areas for global exploration bonus
    global_visited = set()
    
    # Training metrics for analytics
    training_metrics = {
        'episode': [],
        'avg_reward': [],
        'wall_collisions': [],
        'exploration_ratio': [],
        'smoothness': [],
        'epsilon': []
    }
    
    for episode in range(TRAINING_EPISODES):
        # Reset environment
        worm_positions = [(width // 2, height // 2)] * 15
        velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]
        positions_history = []
        episode_reward = 0
        wall_collisions = 0
        smoothness_values = []
        
        # Track visited areas for this episode
        episode_visited = set()
        
        for step in range(STEPS_PER_EPISODE):
            # Get current state
            state = agent.get_state(worm_positions, velocity, (width, height))
            
            # Get action from agent
            action = agent.act(state)
            
            # Update physics (simplified for fast training)
            movement = agent.get_action_movement(action)
            velocity = movement
            head_x, head_y = worm_positions[-1]
            new_x = head_x + velocity[0]
            new_y = head_y + velocity[1]
            
            # Boundary check
            wall_collision = False
            if new_x < 0:
                new_x = 0
                wall_collision = True
            elif new_x > width:
                new_x = width
                wall_collision = True
            if new_y < 0:
                new_y = 0
                wall_collision = True
            elif new_y > height:
                new_y = height
                wall_collision = True
                
            if wall_collision:
                wall_collisions += 1
            
            new_position = (new_x, new_y)
            worm_positions.append(new_position)
            positions_history.append(new_position)
            
            if len(worm_positions) > 15:
                worm_positions.pop(0)
            
            # Calculate reward components
            wall_penalty = -2.0 if wall_collision else 0.0
            
            # Movement smoothness reward
            if len(worm_positions) > 1:
                prev_pos = worm_positions[-2]
                movement_vector = [new_x - prev_pos[0], new_y - prev_pos[1]]
                current_smoothness = math.exp(-abs(math.atan2(movement_vector[1], movement_vector[0])))
                smoothness_values.append(current_smoothness)
            else:
                current_smoothness = 0
            
            # Enhanced exploration rewards
            grid_cell = (int(new_x//50), int(new_y//50))
            episode_visited.add(grid_cell)
            if grid_cell not in global_visited:
                global_visited.add(grid_cell)
                exploration_bonus = 0.5  # Bigger bonus for discovering new areas
            else:
                exploration_bonus = 0.1 if grid_cell not in episode_visited else 0.0
            
            # Distance from center is now a quadratic penalty to strongly discourage corners
            center_x, center_y = width/2, height/2
            distance_from_center = math.sqrt((new_x - center_x)**2 + (new_y - center_y)**2)
            max_distance = math.sqrt((width/2)**2 + (height/2)**2)
            corner_penalty = -0.3 * (distance_from_center / max_distance)**2  # Quadratic penalty
            
            # Add stagnation penalty
            if len(positions_history) > 50:  # Check last 50 positions
                recent_positions = positions_history[-50:]
                unique_cells = len(set((int(p[0]//30), int(p[1]//30)) for p in recent_positions))
                stagnation_penalty = -0.2 * (1 - unique_cells/50)  # Penalize staying in same area
            else:
                stagnation_penalty = 0
            
            # Calculate final reward with new penalties
            reward = (wall_penalty * 1.5 +  # Increased wall penalty
                     current_smoothness * 1.0 +
                     exploration_bonus * 2.0 +  # Increased exploration bonus
                     corner_penalty +
                     stagnation_penalty)
            
            episode_reward += reward
            
            # Get next state
            next_state = agent.get_state(worm_positions, velocity, (width, height))
            
            # Store experience and train
            agent.memory.push(state, action, reward, next_state, False)
            loss = agent.train()
        
        # Calculate episode metrics
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)
            
        # Update analytics metrics
        avg_reward = np.mean(recent_rewards)
        exploration_ratio = len(episode_visited) / ((width // 50) * (height // 50))  # Normalized by total possible cells
        avg_smoothness = np.mean(smoothness_values) if smoothness_values else 0
        
        training_metrics['episode'].append(episode)
        training_metrics['avg_reward'].append(avg_reward)
        training_metrics['wall_collisions'].append(wall_collisions)
        training_metrics['exploration_ratio'].append(exploration_ratio)
        training_metrics['smoothness'].append(avg_smoothness)
        training_metrics['epsilon'].append(agent.epsilon)
        
        # Update analytics with current metrics
        analytics.update_metrics(
            episode=episode,
            metrics={
                'avg_reward': avg_reward,
                'wall_collisions': wall_collisions,
                'exploration_ratio': exploration_ratio,
                'movement_smoothness': avg_smoothness,
                'epsilon': agent.epsilon
            }
        )
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_state(episode, is_best=True)
            print(f"\n New best model saved! Reward: {episode_reward:.2f}")
        
        # Regular save interval
        if episode % SAVE_INTERVAL == 0:
            agent.save_state(episode)
        
        # Update progress and metrics every episode
        avg_recent_reward = np.mean(recent_rewards)
        progress.update(episode + 1)
        
        if episode % PRINT_INTERVAL == 0:
            sys.stdout.write(f' | Avg Reward: {avg_recent_reward:.2f} | Epsilon: {agent.epsilon:.3f}')
            sys.stdout.flush()
        
        # Generate analytics report periodically
        if episode % 50 == 0 and episode > 0:
            report_path = analytics.generate_report(episode)
            print(f"\nGenerated report: {report_path}")

    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\n\nTraining completed!")
    print(f"Total time: {timedelta(seconds=int(elapsed_time))}")
    print(f"Final average reward (last 50 episodes): {np.mean(recent_rewards):.2f}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final epsilon value: {agent.epsilon:.3f}")
    print("\nBest model saved as: models/saved/worm_state_best.pt")
    print("Run 'python app.py --demo' to see the trained worm in action!")

if __name__ == "__main__":
    try:
        fast_training()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Progress has been saved.")
        print("You can still run 'python app.py --demo' to see the current best model!")
    finally:
        pygame.quit()

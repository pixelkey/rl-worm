import pygame
import math
import time
import os
from analytics.metrics import WormAnalytics
from app import WormGame
from agent import WormAgent
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train the worm AI')
parser.add_argument('--demo', action='store_true', help='Run in demo mode using the best trained model')
args = parser.parse_args()

# Initialize game and agent
# State size is now 11: position (2), angle (1), prev_action (1), hunger (1), food_info (2), walls (4)
STATE_SIZE = 11
ACTION_SIZE = 9  # 8 directions + no movement

game = WormGame()
agent = WormAgent(STATE_SIZE, ACTION_SIZE)

# Load the best model if in demo mode
if args.demo:
    agent.epsilon = 0.01  # Very little exploration in demo mode
    print("Running in demo mode with best trained model")

analytics = WormAnalytics()

def train_episode():
    state = game.reset()
    total_reward = 0
    done = False
    steps = 0
    max_steps = 2000
    
    while not done and steps < max_steps:
        action = agent.act(state)
        next_state, reward, done = game.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1
        
        if not args.demo:
            loss = agent.train()
            if loss is not None:
                analytics.update_metrics(steps, {'loss': loss})
        
        if args.demo or steps % 10 == 0:  # Render every frame in demo, every 10th in training
            game.render()
            
    return total_reward, steps

def fast_training():
    print("Starting fast training mode...")
    episode = 0
    target_update_freq = 1000
    save_freq = 100
    
    try:
        while True:
            episode += 1
            total_reward, steps = train_episode()
            
            # Update target network periodically
            if episode % target_update_freq == 0:
                agent.update_target_model()
            
            # Save model periodically
            if episode % save_freq == 0:
                agent.save(episode)
            
            print(f"Episode: {episode}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save(episode)
        game.close()

if __name__ == "__main__":
    try:
        if args.demo:
            print("Running in demo mode...")
            while True:
                train_episode()
        else:
            fast_training()
    except KeyboardInterrupt:
        game.close()

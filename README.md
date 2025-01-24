# AI Worm with Deep Reinforcement Learning

This project implements an intelligent worm that learns to navigate its environment using Deep Reinforcement Learning (DRL). The worm uses a Deep Q-Network (DQN) to develop sophisticated movement patterns and exploration strategies.

## Features

- **Deep Q-Learning Implementation**: Uses PyTorch for neural network training
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support
- **Real-time Visualization**: Pygame-based display of the worm's behavior
- **Analytics Dashboard**: Tracks and visualizes learning progress
- **Fast Training Mode**: Dedicated training script for rapid learning
- **Save/Load System**: Preserves learned behaviors across sessions

## How It Works

### Neural Network Architecture
The worm's brain is a deep neural network with the following structure:
```python
DQN Architecture:
- Input Layer (8 neurons): State information
- Hidden Layer 1 (256 neurons): Pattern recognition
- Hidden Layer 2 (256 neurons): Complex feature processing
- Hidden Layer 3 (128 neurons): Decision refinement
- Output Layer (9 neurons): Movement actions
```

### State Space
The worm perceives its environment through:
- Current position
- Distance to walls
- Movement velocity
- Previous positions (for smoothness)

### Action Space
The worm can choose from 9 actions:
- 8 directional movements (0-7): Corresponding to angles in 45° increments
- No movement (8): Stay in place

### Reward System
The worm learns through a sophisticated reward system:
1. **Exploration Bonus**: +0.5 for discovering new areas
2. **Smoothness Reward**: Based on movement continuity
3. **Wall Avoidance**: -2.0 penalty for wall collisions
4. **Anti-Stagnation**: Penalties for staying in one area
5. **Corner Avoidance**: Quadratic penalty based on distance from center

## Training Process

The training uses several advanced DRL techniques:
- **Experience Replay**: 50,000 memory buffer size
- **Batch Learning**: Processes 2048 experiences per step (4 batches of 512)
- **Target Network**: Updated every 1000 steps for stable learning
- **Epsilon-Greedy Strategy**: 
  - Starts at 1.0 (full exploration)
  - Decays to 0.01 (mostly exploitation)
  - Decay rate of 0.995

## Usage

1. **Demo Mode**:
```bash
python app.py
```
Shows the worm using its best learned behavior

2. **Training Mode**:
```bash
python train.py
```
Starts fast training mode with analytics

3. **Key Controls**:
- ESC: Exit
- SPACE: Pause/Resume
- R: Reset position

## Analytics

The system generates detailed analytics every 50 episodes:
- Learning progress (rewards)
- Movement smoothness
- Exploration coverage
- Wall collision frequency
- Training metrics (epsilon, loss)

## Requirements

- Python 3.12+
- PyTorch with CUDA support
- Pygame
- NumPy
- Pandas (for analytics)
- Plotly (for visualization)

## Project Structure

```
.
├── app.py          # Main visualization application
├── train.py        # Fast training script
├── models/
│   └── dqn.py     # DQN implementation
├── analytics/
│   └── metrics.py  # Analytics and reporting
└── README.md
```

## Future Improvements

- Multi-agent training
- Curriculum learning
- Environment complexity scaling
- Competitive scenarios
- Enhanced sensory inputs
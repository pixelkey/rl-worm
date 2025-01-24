# AI Worm with Deep Reinforcement Learning

This project implements an intelligent worm that learns to navigate its environment using Deep Reinforcement Learning (DRL). The worm uses a Deep Q-Network (DQN) to develop sophisticated movement patterns and exploration strategies.

## Installation

1. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA support
pip install pygame numpy matplotlib pandas plotly seaborn
```

3. **Create required directories**:
```bash
mkdir -p models/saved analytics/reports
```

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

1. **Training Mode**:
```bash
python train.py
```
This starts the training process. The worm will learn to:
- Navigate efficiently
- Avoid walls
- Explore the environment
- Develop smooth movement patterns

2. **Demo Mode**:
```bash
python app.py --demo
```
Shows the worm using its best learned behavior.

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

## Requirements

- Python 3.12+
- PyTorch with CUDA support
- Pygame
- NumPy
- Pandas (for analytics)
- Plotly (for visualization)

## Future Improvements

- Multi-agent training
- Curriculum learning
- Environment complexity scaling
- Competitive scenarios
- Enhanced sensory inputs
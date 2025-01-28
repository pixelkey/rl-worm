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
- **Adaptive Difficulty**: Game difficulty scales with worm length
- **Dynamic Plant System**: Plants grow, mature, and die naturally
- **Emotional Expression System**: Visual feedback of worm's internal state
- **Smooth Movement**: Enhanced rewards for fluid motion
- **Learning Rate Scheduling**: Automatic learning rate adjustment
- **CUDA Optimization**: Improved GPU utilization for faster training
- **Headless Mode**: Support for training without visualization

## How It Works

### Neural Network Architecture
The worm's brain is implemented using PyTorch with the following architecture:
```python
DQN Architecture:
- Input Layer (14 neurons): State information (position, velocity, angle, plants, walls, hunger)
- Hidden Layer 1 (256 neurons, ReLU): Pattern recognition with Kaiming initialization
- Hidden Layer 2 (256 neurons, ReLU): Complex feature processing
- Hidden Layer 3 (128 neurons, ReLU): Decision refinement
- Output Layer (9 neurons): Movement actions (8 directions + no movement)
```

### Advanced Implementation Details
- **Weight Initialization**: Kaiming normal initialization for better gradient flow
- **Optimizer**: Adam optimizer with learning rate 0.0005
- **Learning Rate**: Step LR scheduler (gamma=0.9, step_size=100)
- **Reward Scaling**: Dynamic reward normalization with 20-step window
- **Minimum Std**: 5.0 (increased from 1.0 for more variation)
- **Memory Buffer**: 100,000 experiences
- **Batch Size**: 64 samples (512 in fast training mode)
- **Epsilon**: Starts at 1.0, decays to 0.1, decay rate 0.9998

### Reward System
The worm learns through a sophisticated reward system:

1. **Food and Hunger**:
   - Base food reward: +100.0
   - Hunger multiplier: 2.0x
   - Starvation penalty: -1.5 base rate
   - Shrinking penalty: -25.0

2. **Movement and Exploration**:
   - Smooth movement: +2.0
   - Exploration bonus: +5.0
   - Sharp turns: -0.1
   - Direction changes: -0.05

3. **Growth and Progress**:
   - Growth reward: +50.0
   - Plant spawn mechanics:
     - Base spawn chance: 0.02
     - Min plants: 2, Max plants: 8
     - Spawn cooldown: 60 frames

4. **Safety and Survival**:
   - Wall collision: -50.0
   - Wall proximity: -2.0
   - Extended wall contact: -20.0 base, scaling by 1.2x
   - Danger zone starts at 1.8x head size
   - Wall stay recovery: 0.4

### Adaptive Difficulty
- Hunger system:
  - Max hunger: 1000
  - Base hunger rate: 0.1
  - Hunger gain from plant: 300
  - Shrink threshold: 50% hunger
  - Shrink cooldown: 60 frames

## Maslow's Hierarchy in AI Behavior

The worm's behavior and learning is guided by a reward system based on Maslow's Hierarchy of Needs, creating more realistic and human-like behavior patterns:

### 1. Physiological Needs (Survival)
- **Food & Hunger**: Primary survival drive
  - Exponential rewards (10-30) for eating based on hunger level
  - Severe starvation penalties:
    - -0.1 at 50% hunger (mild discomfort)
    - -0.4 at 25% hunger (increasing distress)
    - -2.25 at 10% hunger (severe penalty)
    - -25.0 at 1% hunger (critical survival state)
  - Emotional expressions show distress when starving

### 2. Safety Needs
- **Collision Avoidance**: -2.0 penalty for wall collisions
- **Movement Safety**: 
  - Penalties up to -1.0 for sharp turns
  - Small rewards (0.1) for smooth, stable movement
  - Expressions reflect distress during unsafe behavior

### 3. Growth & Self-Actualization
- **Growth Rewards**: +8.0 for growing longer
- Only available when basic needs are met (>50% hunger)
- Enhanced happiness expressions when growing while healthy

### 4. Exploration & Discovery
- Small rewards (0.05) for movement and exploration
- Only activated when well-fed (>80% hunger)
- Encourages curiosity once basic needs are satisfied

This hierarchical reward structure ensures the worm:
1. Prioritizes survival above all else
2. Develops safe movement patterns
3. Pursues growth only when healthy
4. Explores its environment when comfortable

The worm's facial expressions provide visual feedback of its current state in the hierarchy, from distress when starving to contentment during healthy growth.

## Expression and Animation System

The worm features a sophisticated facial animation system that brings personality to the AI agent:

#### Eye System
- Dynamic pupil movement with smooth interpolation (speed: 0.15)
- Natural blinking system with random intervals (3-6 seconds)
- Smooth blink animations (0.3s duration)
- Minimum 2.0s between blinks
- Eye convergence system (max 40% inward convergence)
- Detailed eye anatomy:
  - White sclera
  - Black pupils
  - Dynamic pupil positioning
  - Size scales with head size (25% of head size)

#### Expression System
- Three expression states: smile (1), neutral (0), frown (-1)
- Smooth expression transitions
- Expression magnitude control
- Configurable hold times for expressions
- Base expression change speed: 2.0 (adjustable by magnitude)

#### Animation Features
- Smooth movement interpolation
- Dynamic head rotation
- Fluid body segment following
- Responsive wall collision reactions
- Hunger state visual feedback

## Training Process

The training uses several advanced DRL techniques:
- **Experience Replay**: 
  - 50,000 memory buffer size with normalized rewards
  - Reward clipping to [-1.0, 1.0] range
- **Batch Learning**: 
  - Regular mode: 64 experiences per batch
  - Fast training mode: 512 experiences per batch
  - Multiple batches processed for stable learning
- **Target Network**: Updated every 1000 steps for stable learning
- **Epsilon-Greedy Strategy**: 
  - Starts at 1.0 (full exploration)
  - Regular mode:
    - Decays to 0.1 (increased minimum exploration)
    - Slower decay rate of 0.9998
  - Fast training mode:
    - Decays to 0.01
    - Decay rate of 0.9995

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
- PyTorch 2.0+ with CUDA support
- Pygame 2.5+
- NumPy
- Pandas (for analytics)
- Plotly (for visualization)
- Seaborn (for additional visualizations)

## Future Improvements

- Multi-agent training
- Curriculum learning
- Environment complexity scaling
- Competitive scenarios
- Enhanced sensory inputs
# ML Agent Configuration
STATE_SIZE = 15  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), plant_value (1), walls (4), hunger (1)
ACTION_SIZE = 9  # 8 directions + no movement

# Training Parameters
TRAINING_EPISODES = 1000
STARTING_STEPS = 1000  # Starting number of steps
MAX_STEPS = 6000  # Maximum steps allowed
STEPS_INCREMENT = 50  # Changed from 200 to 50 steps per episode
PERFORMANCE_THRESHOLD = -50  # More lenient threshold

# Save and print intervals
SAVE_INTERVAL = 10  # Save model every N episodes
PRINT_INTERVAL = 1  # Print metrics every N episodes

# Base Reward Constants
REWARD_FOOD_BASE = 350.0  # Increased to make food more rewarding
REWARD_GROWTH = 300.0  # Reward for growing
PENALTY_WALL = -80.0  # Keep strong wall collision penalty
PENALTY_DEATH = -500.0  # Penalty for dying (starvation or no segments)

# Additional Reward Modifiers
REWARD_FOOD_HUNGER_SCALE = 2.0  # Scale factor for food reward based on hunger
REWARD_SMOOTH_MOVEMENT = 1.5  # Reward for smooth movement
REWARD_EXPLORATION = 20.0  # Reward for exploring new areas

# Additional Penalties
PENALTY_WALL_STAY = -5.0  # Keep strong wall stay penalty
PENALTY_SHARP_TURN = -0.1  # Small penalty for sharp turns to encourage smooth movement
PENALTY_SHRINK = -15.0  # Penalty for shrinking
PENALTY_DANGER_ZONE = -2.0  # Keep as is
PENALTY_STARVATION_BASE = -1.5  # Base penalty for starvation

# Display Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60  # Frames per second

# ML Agent Configuration
STATE_SIZE = 15  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), plant_value (1), walls (4), hunger (1)
ACTION_SIZE = 9  # 8 directions + no movement

# Training Parameters
TRAINING_EPISODES = 1000
STARTING_STEPS = 1000  # Starting number of steps
MAX_STEPS = 6000  # Maximum steps allowed
STEPS_INCREMENT = 50  # Changed from 200 to 50 steps per episode
PERFORMANCE_THRESHOLD = -50  # More lenient threshold

# Save and Model Paths
MODEL_DIR = "models/saved"
MODEL_PATH = "models/saved/worm_model.pth"
CHECKPOINT_PATH = "models/saved/checkpoint.json"

# Save and print intervals
SAVE_INTERVAL = 10  # Save model every N episodes
PRINT_INTERVAL = 1  # Print metrics every N episodes

# Level Parameters
MIN_STEPS = 6000  # Minimum steps per level
MAX_LEVEL_STEPS = 20000  # Maximum steps per level
LEVEL_STEPS_INCREMENT = 1000  # Steps increase per level

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
PENALTY_DIRECTION_CHANGE = -0.05  # Small penalty for changing direction to encourage smooth movement
PENALTY_SHRINK = -15.0  # Penalty for shrinking
PENALTY_DANGER_ZONE = -2.0  # Keep as is
PENALTY_STARVATION_BASE = -1.5  # Base penalty for starvation

# Worm Properties
MAX_SEGMENTS = 30  # Maximum number of body segments
MIN_SEGMENTS = 2  # Minimum number of body segments

# Hunger Mechanics
MAX_HUNGER = 1000  # Maximum hunger value
BASE_HUNGER_RATE = 0.1  # Rate at which hunger increases
HUNGER_GAIN_FROM_PLANT = 300  # Hunger restored when eating a plant
SHRINK_HUNGER_THRESHOLD = 0.5  # Threshold at which worm starts shrinking

# Plant Properties
MIN_PLANTS = 2  # Minimum number of plants in environment
MAX_PLANTS = 8  # Maximum number of plants in environment
PLANT_SPAWN_CHANCE = 0.02  # Base chance of spawning a new plant

# Display Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60  # Frames per second

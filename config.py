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
REWARD_FOOD_BASE = 500.0  # Increased from 350 to make food more rewarding
REWARD_GROWTH = 400.0  # Increased from 300
PENALTY_WALL = -200.0  # Increased from -80 to make walls more punishing
PENALTY_DEATH = -1000.0  # Increased from -500 to make death more punishing

# Additional Reward Modifiers
REWARD_FOOD_HUNGER_SCALE = 2.0  # Scale factor for food reward based on hunger
REWARD_SMOOTH_MOVEMENT = 2.0  # Increased from 1.5
REWARD_EXPLORATION = 50.0  # Increased from 20 to encourage exploration
REWARD_SURVIVAL = 0.1  # Small positive reward for each step survived

# Additional Penalties
PENALTY_WALL_STAY = -20.0  # Increased from -5 to discourage wall hugging
PENALTY_SHARP_TURN = -0.5  # Increased from -0.1
PENALTY_DIRECTION_CHANGE = -0.2  # Increased from -0.05
PENALTY_SHRINK = -50.0  # Increased from -15
PENALTY_DANGER_ZONE = -10.0  # Increased from -2 to make wall proximity more punishing
PENALTY_STARVATION_BASE = -2.0  # Increased from -1.5

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

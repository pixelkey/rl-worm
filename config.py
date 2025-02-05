# ML Agent Configuration
STATE_SIZE = 15  # position (2), velocity (2), angle (1), angular_vel (1), plant info (3), plant_value (1), walls (4), hunger (1)
ACTION_SIZE = 9  # 8 directions + no movement

# Training Parameters
TRAINING_EPISODES = 1000
STARTING_STEPS = 500
MAX_STEPS = 2000
STEPS_INCREMENT = 100
PERFORMANCE_THRESHOLD = 0.8

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 0.995

# Exploration Parameters
EPSILON = 0.2  # Starting exploration rate
EPSILON_MIN = 0.05  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Slower decay for more stable learning

# Memory and Learning
MEMORY_SIZE = 100000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor
TARGET_UPDATE = 10  # Update target network every N episodes

# Save and Model Paths
MODEL_DIR = "models/saved"
MODEL_PATH = "models/saved/worm_model.pth"
CHECKPOINT_PATH = "models/saved/checkpoint.json"

# Save and print intervals
SAVE_INTERVAL = 100  # Save model every N episodes
PRINT_INTERVAL = 1  # Print metrics every N episodes

# Level Parameters
MIN_STEPS = 100
MAX_LEVEL_STEPS = 2000
LEVEL_STEPS_INCREMENT = 100

# Base Reward Constants
REWARD_FOOD_BASE = 500.0  # Base reward for eating food
REWARD_GROWTH = 400.0  # Additional reward for growing
PENALTY_WALL = -200.0  # Penalty for hitting wall
PENALTY_DEATH = -1000.0  # Penalty for dying

# Additional Reward Modifiers
REWARD_FOOD_HUNGER_SCALE = 2.0  # Scale food reward based on hunger
REWARD_SMOOTH_MOVEMENT = 2.0  # Reward for smooth movement
REWARD_EXPLORATION = 50.0  # Reward for exploring new areas
REWARD_SURVIVAL = 0.1  # Small reward for each step survived
REWARD_DISTANCE = 1.0  # Reward for distance traveled per step

# Additional Penalties
PENALTY_WALL_STAY = -20.0  # Penalty for staying near wall
PENALTY_SHARP_TURN = -5.0  # Penalty for sharp turns
PENALTY_SHRINK = -50.0  # Penalty for shrinking
PENALTY_DANGER_ZONE = -10.0  # Penalty for being near wall
PENALTY_STARVATION_BASE = -20.0  # Base penalty for starvation
PENALTY_DIRECTION_CHANGE = -2.0  # Penalty for changing direction

# Worm Properties
MAX_SEGMENTS = 20  # Maximum number of body segments
MIN_SEGMENTS = 3  # Minimum number of body segments

# Hunger Mechanics
MAX_HUNGER = 1000  # Maximum hunger value
BASE_HUNGER_RATE = 0.1  # Rate at which hunger increases
HUNGER_GAIN_FROM_PLANT = 500  # Hunger restored when eating a plant
SHRINK_HUNGER_THRESHOLD = 0  # Threshold at which worm starts shrinking

# Plant Properties
MIN_PLANTS = 3  # Minimum number of plants in environment
MAX_PLANTS = 5  # Maximum number of plants in environment
PLANT_SPAWN_CHANCE = 0.02  # Base chance of spawning a new plant

# Display Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60  # Frames per second

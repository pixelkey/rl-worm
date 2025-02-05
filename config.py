# ML Agent Configuration
STATE_SIZE = 20  # Updated from 15: now includes 12 values from plants + 3 for direction/speed + 4 wall distances + 1 hunger level = 20
ACTION_SIZE = 9  # 8 directions + no movement

# Training Parameters
TRAINING_EPISODES = 1000
STARTING_STEPS = 800
MAX_STEPS = 6000
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
REWARD_FOOD_BASE = 0.5    # Base reward for eating food
REWARD_GROWTH = 0.4       # Additional reward for growing
PENALTY_WALL = -1.0       # Penalty for hitting wall
PENALTY_DEATH = -2.0      # Penalty for dying

# Additional Reward Modifiers
REWARD_FOOD_HUNGER_SCALE = 1.5  # Multiplier for food reward based on hunger
REWARD_SMOOTH_MOVEMENT = 0.05   # Reward for smooth movement
REWARD_EXPLORATION = 0.05       # Reward for exploring new areas
REWARD_SURVIVAL = 0.1  # Small reward for each step survived
REWARD_DISTANCE = 0.01          # Small continuous reward for distance traveled

# Additional Penalties
PENALTY_WALL_STAY = -0.5        # Penalty for staying near wall
PENALTY_SHRINK = -0.2           # Penalty for shrinking
PENALTY_DANGER_ZONE = -0.2      # Penalty for being near wall
PENALTY_STARVATION_BASE = -0.5  # Penalty for starvation
PENALTY_SHARP_TURN = -0.001       # Penalty for sharp turns
PENALTY_DIRECTION_CHANGE = -0.002  # Penalty for changing direction

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

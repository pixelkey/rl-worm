import pygame
import math
import random

class PlantType:
    def __init__(self, name, characteristics):
        self.name = name
        self.max_branch_depth = characteristics.get('max_branch_depth', 3)
        self.base_color = characteristics.get('base_color', (20, 150, 20))
        self.leaf_color = characteristics.get('leaf_color', None)  # If None, use base_color
        self.flower_colors = characteristics.get('flower_colors', [])  # List of possible flower colors
        self.flower_type = characteristics.get('flower_type', 'simple')  # Type of flower to draw
        self.stem_color = characteristics.get('stem_color', None)  # If None, use base_color
        self.leaf_type = characteristics.get('leaf_type', 'normal')
        self.stem_type = characteristics.get('stem_type', 'straight')
        self.growth_speed = characteristics.get('growth_speed', 1.0)
        self.branch_density = characteristics.get('branch_density', (0.6, 1.0))
        self.mature_height_multiplier = characteristics.get('mature_height_multiplier', 1.0)
        self.has_flowers = characteristics.get('has_flowers', False)
        self.flower_probability = characteristics.get('flower_probability', 0.3)
        self.petal_count = characteristics.get('petal_count', 5)

# Define different plant types
PLANT_TYPES = {
    'fern': PlantType('fern', {
        'base_color': (20, 150, 20),
        'leaf_color': (30, 160, 30),
        'stem_color': (60, 120, 30),
        'max_branch_depth': 4,
        'leaf_type': 'feather',
        'stem_type': 'curved',
        'growth_speed': 1.2,
        'branch_density': (0.8, 0.9),
        'mature_height_multiplier': 0.8
    }),
    'bush': PlantType('bush', {
        'base_color': (20, 140, 20),
        'leaf_color': (40, 150, 40),
        'stem_color': (80, 100, 40),
        'max_branch_depth': 3,
        'leaf_type': 'round',
        'stem_type': 'straight',
        'growth_speed': 0.9,
        'branch_density': (0.7, 1.0),
        'mature_height_multiplier': 0.7
    }),
    'purple_flower': PlantType('purple_flower', {
        'base_color': (150, 50, 150),
        'leaf_color': (40, 160, 40),
        'stem_color': (60, 140, 60),
        'flower_colors': [(200, 100, 200), (180, 80, 180), (220, 120, 220)],  # Purple variations
        'flower_type': 'star',
        'max_branch_depth': 3,
        'leaf_type': 'pointed',
        'stem_type': 'straight',
        'growth_speed': 1.1,
        'branch_density': (0.5, 0.7),
        'mature_height_multiplier': 1.2,
        'has_flowers': True,
        'flower_probability': 0.4,
        'petal_count': 5
    }),
    'daisy': PlantType('daisy', {
        'base_color': (60, 160, 60),
        'leaf_color': (40, 150, 40),
        'stem_color': (80, 140, 80),
        'flower_colors': [(255, 255, 255), (255, 250, 240), (255, 245, 230)],  # White variations
        'flower_type': 'daisy',
        'max_branch_depth': 3,
        'leaf_type': 'pointed',
        'stem_type': 'straight',
        'growth_speed': 1.0,
        'branch_density': (0.4, 0.6),
        'mature_height_multiplier': 1.0,
        'has_flowers': True,
        'flower_probability': 0.5,
        'petal_count': 12
    }),
    'rose': PlantType('rose', {
        'base_color': (60, 140, 60),
        'leaf_color': (30, 130, 30),
        'stem_color': (80, 100, 40),
        'flower_colors': [(220, 50, 50), (200, 40, 40), (180, 30, 30)],  # Red variations
        'flower_type': 'rose',
        'max_branch_depth': 3,
        'leaf_type': 'pointed',
        'stem_type': 'straight',
        'growth_speed': 0.9,
        'branch_density': (0.6, 0.8),
        'mature_height_multiplier': 1.1,
        'has_flowers': True,
        'flower_probability': 0.3,
        'petal_count': 20
    }),
    'sunflower': PlantType('sunflower', {
        'base_color': (80, 160, 80),
        'leaf_color': (60, 150, 60),
        'stem_color': (100, 140, 60),
        'flower_colors': [(255, 200, 0), (255, 180, 0), (255, 160, 0)],  # Yellow variations
        'flower_type': 'sunflower',
        'max_branch_depth': 2,
        'leaf_type': 'pointed',
        'stem_type': 'straight',
        'growth_speed': 1.2,
        'branch_density': (0.3, 0.5),
        'mature_height_multiplier': 1.5,
        'has_flowers': True,
        'flower_probability': 0.8,
        'petal_count': 34
    }),
    'vine': PlantType('vine', {
        'base_color': (30, 160, 30),
        'leaf_color': (50, 180, 50),
        'stem_color': (40, 130, 40),
        'max_branch_depth': 4,
        'leaf_type': 'heart',
        'stem_type': 'wavy',
        'growth_speed': 1.3,
        'branch_density': (0.4, 0.6),
        'mature_height_multiplier': 1.4
    }),
    'succulent': PlantType('succulent', {
        'base_color': (100, 160, 100),
        'leaf_color': (120, 180, 120),
        'stem_color': (90, 140, 90),
        'max_branch_depth': 2,
        'leaf_type': 'thick',
        'stem_type': 'thick',
        'growth_speed': 0.7,
        'branch_density': (0.9, 1.0),
        'mature_height_multiplier': 0.5
    }),
    'lotus': PlantType('lotus', {
        'base_color': (50, 160, 50),
        'leaf_color': (40, 140, 40),
        'stem_color': (70, 130, 70),
        'flower_colors': [(255, 192, 203), (255, 182, 193), (255, 172, 183)],  # Pink variations
        'flower_type': 'lotus',
        'max_branch_depth': 2,
        'leaf_type': 'round',
        'stem_type': 'curved',
        'growth_speed': 0.9,
        'branch_density': (0.3, 0.5),
        'mature_height_multiplier': 0.8,
        'has_flowers': True,
        'flower_probability': 0.6,
        'petal_count': 12
    }),
    'orchid': PlantType('orchid', {
        'base_color': (60, 150, 60),
        'leaf_color': (40, 130, 40),
        'stem_color': (80, 120, 80),
        'flower_colors': [
            (216, 191, 216),  # Light purple
            (238, 130, 238),  # Violet
            (255, 240, 245),  # White-pink
            (255, 182, 193)   # Light pink
        ],
        'flower_type': 'orchid',
        'max_branch_depth': 3,
        'leaf_type': 'pointed',
        'stem_type': 'curved',
        'growth_speed': 0.8,
        'branch_density': (0.4, 0.6),
        'mature_height_multiplier': 1.1,
        'has_flowers': True,
        'flower_probability': 0.4,
        'petal_count': 5
    }),
    'cherry_blossom': PlantType('cherry_blossom', {
        'base_color': (90, 120, 90),
        'leaf_color': (60, 150, 60),
        'stem_color': (139, 69, 19),  # Saddle brown
        'flower_colors': [
            (255, 223, 223),  # Very light pink
            (255, 218, 218),
            (255, 192, 203)   # Pink
        ],
        'flower_type': 'cherry_blossom',
        'max_branch_depth': 4,
        'leaf_type': 'pointed',
        'stem_type': 'branching',
        'growth_speed': 1.0,
        'branch_density': (0.7, 0.9),
        'mature_height_multiplier': 1.4,
        'has_flowers': True,
        'flower_probability': 0.7,
        'petal_count': 5
    }),
    'tropical': PlantType('tropical', {
        'base_color': (34, 139, 34),  # Forest green
        'leaf_color': (50, 205, 50),  # Lime green
        'stem_color': (85, 107, 47),  # Dark olive green
        'flower_colors': [
            (255, 140, 0),   # Dark orange
            (255, 165, 0),   # Orange
            (255, 69, 0)     # Red-orange
        ],
        'flower_type': 'bird_of_paradise',
        'max_branch_depth': 3,
        'leaf_type': 'tropical',
        'stem_type': 'thick',
        'growth_speed': 1.2,
        'branch_density': (0.5, 0.7),
        'mature_height_multiplier': 1.6,
        'has_flowers': True,
        'flower_probability': 0.3,
        'petal_count': 6
    }),
    'cactus': PlantType('cactus', {
        'base_color': (50, 205, 50),  # Lime green
        'leaf_color': (34, 139, 34),  # Forest green
        'stem_color': (0, 100, 0),    # Dark green
        'flower_colors': [
            (255, 192, 203),  # Pink
            (255, 0, 0),      # Red
            (255, 255, 0)     # Yellow
        ],
        'flower_type': 'cactus_flower',
        'max_branch_depth': 2,
        'leaf_type': 'spike',
        'stem_type': 'thick',
        'growth_speed': 0.5,
        'branch_density': (0.2, 0.4),
        'mature_height_multiplier': 0.9,
        'has_flowers': True,
        'flower_probability': 0.2,
        'petal_count': 12
    }),
    'bamboo': PlantType('bamboo', {
        'base_color': (50, 205, 50),  # Lime green
        'leaf_color': (144, 238, 144),  # Light green
        'stem_color': (85, 107, 47),    # Dark olive green
        'max_branch_depth': 5,
        'leaf_type': 'bamboo',
        'stem_type': 'segmented',
        'growth_speed': 1.5,
        'branch_density': (0.3, 0.5),
        'mature_height_multiplier': 2.0,
        'has_flowers': False
    }),
}

class Plant:
    def __init__(self, x, y, game_size):
        self.x = x
        self.y = y
        self.base_size = int(game_size/40)  # Base size for scaling
        
        # Randomly select a plant type
        self.plant_type = PLANT_TYPES[random.choice(list(PLANT_TYPES.keys()))]
        
        # Adjust characteristics based on plant type
        self.max_lifetime = int(800 * self.plant_type.growth_speed)
        self.lifetime = self.max_lifetime
        self.state = 'seed'
        self.growth_stage = 0.0
        self.stem_height = 0
        self.leaf_size = 0
        self.base_color = self.plant_type.base_color
        self.leaf_color = self.plant_type.leaf_color if self.plant_type.leaf_color else self.base_color
        self.stem_color = self.plant_type.stem_color if self.plant_type.stem_color else self.base_color
        self.color = self.base_color
        
        # Generate unique characteristics for this plant
        self.max_branch_depth = self.plant_type.max_branch_depth
        self.branch_density = random.uniform(*self.plant_type.branch_density)
        self.branch_variation = random.uniform(0.8, 1.2)
        self.leaf_density = random.uniform(0.7, 1.0)
        self.stem_waviness = random.uniform(0.8, 1.2)
        
        # Random color variation while keeping within natural range
        self.color_variation = random.uniform(-20, 20)
        r, g, b = self.color
        self.color = (
            max(0, min(255, r + self.color_variation)),
            max(0, min(255, g + self.color_variation)),
            max(0, min(255, b + self.color_variation))
        )
        
        # Pre-generate all random variations for this plant
        self.branch_patterns = []
        self.leaf_patterns = []
        self.branch_growth = [0.0] * (self.max_branch_depth + 1)
        self.generate_patterns(self.max_branch_depth)
        
        # Generate unique characteristics for this plant
        self.max_branch_depth = random.randint(2, 3)  # How many levels of branching
        self.branch_density = random.uniform(0.6, 1.0)  # How likely branches are to split
        self.branch_variation = random.uniform(0.8, 1.2)  # Multiplier for branch sizes
        self.leaf_density = random.uniform(0.7, 1.0)  # How many leaves to generate
        self.stem_waviness = random.uniform(0.8, 1.2)  # How much the stem waves
        
        # Random color variation
        self.color_variation = random.uniform(-20, 20)  # Each plant slightly different color
        
        # Pre-generate all random variations for this plant
        self.branch_patterns = []  # Will store [angle, length, ctrl_point] for each branch
        self.leaf_patterns = []    # Will store [size, position] for each potential leaf
        self.generate_patterns(self.max_branch_depth)
        
        self.branch_growth = [0.0] * (self.max_branch_depth + 1)  # List of growth stages for each branch level
        
        # Store consistent colors for this plant instance
        self.instance_flower_color = random.choice(self.plant_type.flower_colors) if self.plant_type.flower_colors else None
        self.instance_leaf_color = self.plant_type.leaf_color or self.plant_type.base_color
        self.instance_stem_color = self.plant_type.stem_color or self.plant_type.base_color

    def generate_patterns(self, max_depth, parent_angle=90, depth=0):
        """Pre-generate all random patterns for this plant"""
        if depth >= max_depth:
            return
            
        num_branches = random.randint(2, 3)
        branch_angles = []
        
        # Generate asymmetric branch angles
        for i in range(num_branches):
            if i == 0:
                angle_var = random.uniform(20, 40)
                branch_angles.append(angle_var)
            else:
                angle_var = random.uniform(15, 35)
                sign = 1 if random.random() < 0.5 else -1
                branch_angles.append(angle_var * sign)
        
        # Sort angles to prevent crossing
        branch_angles.sort()
        
        # Store patterns for each branch
        for angle in branch_angles:
            variation = random.uniform(0.95, 1.05)
            ctrl_offset = (random.uniform(-5, 5), random.uniform(-5, 5))
            length_var = random.uniform(0.9, 1.1)
            
            self.branch_patterns.append({
                'angle': angle,
                'variation': variation,
                'ctrl_offset': ctrl_offset,
                'length_var': length_var,
                'depth': depth,
                'has_leaf': random.random() < self.leaf_density,
                'leaf_size_var': random.uniform(0.8, 1.2) if random.random() < self.leaf_density else 1.0,
                'has_flower': random.random() < self.plant_type.flower_probability if self.plant_type.has_flowers else False
            })
            
            # Recursively generate patterns for sub-branches
            if random.random() < self.branch_density:
                self.generate_patterns(max_depth, angle + parent_angle, depth + 1)

    def update(self):
        """Update plant state based on lifetime"""
        self.lifetime -= 1
        life_ratio = self.lifetime / self.max_lifetime
        
        # Update state and growth based on lifetime
        if life_ratio > 0.9:  # First 10% - seed
            self.state = 'seed'
            stage_progress = (1.0 - life_ratio) / 0.1  # 0.0 to 1.0 during seed stage
            self.growth_stage = 0.1 * stage_progress  # Max 0.1 during seed stage
            self.stem_height = int(self.base_size * 0.2)
            self.leaf_size = 0
            self.color = (30, 110, 20)  # Dark green seed
            self.branch_growth = [0.0] * (self.max_branch_depth + 1)
        elif life_ratio > 0.7:  # Next 20% - sprouting
            self.state = 'sprouting'
            stage_progress = (0.9 - life_ratio) / 0.2  # 0.0 to 1.0 during sprouting
            self.growth_stage = 0.1 + (0.3 * stage_progress)  # 0.1 to 0.4 during sprouting
            self.stem_height = int(self.base_size * (0.2 + stage_progress * 1.8))
            self.leaf_size = int(self.base_size * stage_progress * 0.5)
            green = int(110 + 70 * stage_progress)
            self.color = (20, green, 10)
            self.branch_growth = [stage_progress] + [0.0] * self.max_branch_depth
        elif life_ratio > 0.4:  # Next 30% - growing/mature
            self.state = 'mature'
            stage_progress = (0.7 - life_ratio) / 0.3  # 0.0 to 1.0 during mature stage
            self.growth_stage = 0.4 + (0.6 * stage_progress)  # 0.4 to 1.0 during mature stage
            self.stem_height = int(self.base_size * 2)
            self.leaf_size = self.base_size
            self.color = (20, 180, 10)  # Bright healthy green
            
            # Calculate growth progress for each branch level
            growth_per_level = 0.25  # Each level takes 25% of the growth time to appear
            self.branch_growth = []  # List of growth stages for each branch level
            
            # Main stem is always fully grown in mature stage
            self.branch_growth.append(1.0)
            
            # Calculate growth for each branch level
            for depth in range(1, self.max_branch_depth + 1):
                level_start = (depth - 1) * growth_per_level
                level_progress = max(0.0, min(1.0, (stage_progress - level_start) / growth_per_level))
                self.branch_growth.append(level_progress)
                
        elif life_ratio > 0.1:  # Next 30% - starting to wilt
            self.state = 'wilting'
            stage_progress = (life_ratio - 0.1) / 0.3  # 1.0 to 0.0 during wilting
            self.growth_stage = 0.4 * stage_progress  # 0.4 to 0.0 during wilting
            self.branch_growth = [1.0] * (self.max_branch_depth + 1)  # All branches fully grown
            self.stem_height = int(self.base_size * 2)
            self.leaf_size = int(self.base_size * stage_progress)
            green = int(180 * stage_progress)
            brown = int(120 * (1 - stage_progress))
            self.color = (brown, green, 0)
        else:  # Final 10% - nearly dead
            self.state = 'dying'
            stage_progress = life_ratio / 0.1  # 1.0 to 0.0 during dying
            self.growth_stage = 0.1 * stage_progress  # 0.1 to 0.0 during dying
            self.branch_growth = [1.0] * (self.max_branch_depth + 1)  # All branches fully grown
            self.stem_height = int(self.base_size * (1 + stage_progress))
            self.leaf_size = int(self.base_size * stage_progress)
            self.color = (100, int(60 * stage_progress), 0)  # Brown
            
        return self.lifetime > 0

    def get_bounding_box(self):
        """Get the bounding box for collision detection"""
        # Calculate the actual visible height of the plant based on its growth stage
        max_height = self.stem_height * self.plant_type.mature_height_multiplier
        current_height = max_height * self.growth_stage
        
        # Calculate width based on the actual spread of branches and leaves
        spread_factor = 0.6  # Plants typically spread about 60% of their height
        width = max(self.base_size, current_height * spread_factor)  # At least as wide as base
        
        # Position the box with its bottom center at the plant's base (x, y)
        return pygame.Rect(
            self.x - width/2,  # Left edge
            self.y - current_height,  # Top edge
            width,  # Width
            current_height  # Height
        )

    def draw_fractal_branch(self, surface, start, angle, length, width, depth=0, max_depth=2):
        """Recursively draw a fractal branch with leaves"""
        if depth > max_depth or length < 2:
            return
            
        # Get the pattern for this branch level
        pattern_index = 0
        current_pattern = None
        for i, pattern in enumerate(self.branch_patterns):
            if pattern['depth'] == depth:
                current_pattern = pattern
                pattern_index = i
                break
                
        if current_pattern is None:
            return
            
        # Get growth progress for this branch level
        growth_progress = self.branch_growth[depth]
        if growth_progress <= 0:
            return
            
        # Calculate end point using stored variation and growth progress
        variation = current_pattern['variation']
        current_length = length * growth_progress
        end_x = start[0] + math.cos(math.radians(angle)) * current_length * variation
        end_y = start[1] - math.sin(math.radians(angle)) * current_length * variation
        end = (end_x, end_y)
        
        # Draw the branch
        if depth == 0:  # Main stem
            points = []
            num_points = 8
            time_factor = pygame.time.get_ticks() / 1000.0
            wave_offset = math.sin(time_factor * 2) * 2 * self.growth_stage * self.stem_waviness
            
            for i in range(num_points):
                t = i / (num_points - 1)
                wave = math.sin(t * math.pi + time_factor) * wave_offset
                x = start[0] + t * (end[0] - start[0]) + wave
                y = start[1] + t * (end[1] - start[1])
                points.append((x, y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, self.instance_stem_color, False, points, width)
        else:  # Sub-branches
            # Use stored control point offset
            ctrl_offset = current_pattern['ctrl_offset']
            mid_x = (start[0] + end[0])/2 + ctrl_offset[0] * growth_progress
            mid_y = (start[1] + end[1])/2 + ctrl_offset[1] * growth_progress
            ctrl_point = (mid_x, mid_y)
            
            points = []
            for t in range(10):
                t = t / 9
                # Quadratic bezier for curved branches
                x = (1-t)*((1-t)*start[0] + t*ctrl_point[0]) + t*((1-t)*ctrl_point[0] + t*end[0])
                y = (1-t)*((1-t)*start[1] + t*ctrl_point[1]) + t*((1-t)*ctrl_point[1] + t*end[1])
                points.append((x, y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, self.instance_stem_color, False, points, max(1, int(width * growth_progress)))
        
        # Draw leaves using stored pattern, only if branch is grown enough
        if current_pattern['has_leaf'] and self.state not in ['seed', 'sprouting'] and growth_progress > 0.5:
            leaf_size = self.leaf_size * (0.7 ** depth) * current_pattern['leaf_size_var'] * min(1.0, growth_progress * 2 - 1)
            if leaf_size > 0:
                self.draw_fractal_leaf(surface, end, angle, leaf_size)
        
        # Draw flowers using stored pattern, only if branch is grown enough
        if current_pattern['has_flower'] and self.state not in ['seed', 'sprouting'] and growth_progress > 0.5:
            flower_size = self.leaf_size * (0.7 ** depth) * min(1.0, growth_progress * 2 - 1)
            if flower_size > 0:
                self.draw_flower(surface, end, angle, flower_size)
        
        # Calculate new length and width for sub-branches
        new_length = length * 0.7 * self.branch_variation
        new_width = max(1, int(width * 0.7))
        
        # Draw sub-branches using stored patterns
        if self.state not in ['seed', 'sprouting']:
            next_depth_patterns = [p for p in self.branch_patterns if p['depth'] == depth + 1]
            
            for pattern in next_depth_patterns:
                new_angle = angle + pattern['angle']
                if self.state == 'wilting':
                    # Droop the branches when wilting
                    droop = (1 - self.growth_stage) * 30
                    new_angle -= droop
                
                self.draw_fractal_branch(surface, end, new_angle, 
                                       new_length * pattern['length_var'],
                                       new_width, depth + 1, max_depth)

    def draw_flower(self, surface, pos, angle, size):
        """Draw a flower at the end of a branch"""
        if size < 2 or not self.plant_type.has_flowers:
            return
            
        flower_color = self.instance_flower_color
        petal_size = size * 0.8
        num_petals = self.plant_type.petal_count
        
        if self.plant_type.flower_type == 'star':
            # Star-shaped flower with pointed petals
            center_size = size * 0.2
            pygame.draw.circle(surface, (255, 220, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            for i in range(num_petals):
                petal_angle = angle + (i * (360 / num_petals))
                tip_x = pos[0] + math.cos(math.radians(petal_angle)) * petal_size
                tip_y = pos[1] - math.sin(math.radians(petal_angle)) * petal_size
                left_x = pos[0] + math.cos(math.radians(petal_angle - 20)) * (petal_size * 0.3)
                left_y = pos[1] - math.sin(math.radians(petal_angle - 20)) * (petal_size * 0.3)
                right_x = pos[0] + math.cos(math.radians(petal_angle + 20)) * (petal_size * 0.3)
                right_y = pos[1] - math.sin(math.radians(petal_angle + 20)) * (petal_size * 0.3)
                
                pygame.draw.polygon(surface, flower_color, [
                    (pos[0], pos[1]),
                    (left_x, left_y),
                    (tip_x, tip_y),
                    (right_x, right_y)
                ])
                
        elif self.plant_type.flower_type == 'daisy':
            # Daisy with rounded petals and yellow center
            center_size = size * 0.3
            pygame.draw.circle(surface, (255, 220, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            for i in range(num_petals):
                petal_angle = angle + (i * (360 / num_petals))
                base_x = pos[0] + math.cos(math.radians(petal_angle)) * center_size
                base_y = pos[1] - math.sin(math.radians(petal_angle)) * center_size
                tip_x = pos[0] + math.cos(math.radians(petal_angle)) * petal_size
                tip_y = pos[1] - math.sin(math.radians(petal_angle)) * petal_size
                
                # Draw oval-shaped petal
                points = []
                for t in range(6):
                    t = t / 5
                    radius = petal_size * (0.3 + t * 0.7)
                    curve = math.sin(t * math.pi) * size * 0.15
                    petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                    petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                    petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                    petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                    points.append((petal_x, petal_y))
                
                if len(points) >= 3:
                    pygame.draw.polygon(surface, flower_color, points)
                    
        elif self.plant_type.flower_type == 'rose':
            # Rose with layered petals
            center_size = size * 0.2
            for layer in range(3):  # Multiple layers of petals
                layer_size = petal_size * (0.6 + layer * 0.2)
                layer_petals = num_petals - layer * 4
                layer_color = (
                    max(0, min(255, flower_color[0] - layer * 20)),
                    max(0, min(255, flower_color[1] - layer * 20)),
                    max(0, min(255, flower_color[2] - layer * 20))
                )
                
                for i in range(layer_petals):
                    petal_angle = angle + (i * (360 / layer_petals)) + layer * 10
                    # Draw curved petal
                    points = []
                    for t in range(8):
                        t = t / 7
                        radius = layer_size * (0.3 + t * 0.7)
                        curve = math.sin(t * math.pi) * size * 0.2
                        petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                        petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                        petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                        petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                        points.append((petal_x, petal_y))
                    
                    if len(points) >= 3:
                        pygame.draw.polygon(surface, layer_color, points)
                        
        elif self.plant_type.flower_type == 'sunflower':
            # Sunflower with seed pattern center
            center_size = size * 0.4
            # Draw dark center with seed pattern
            pygame.draw.circle(surface, (101, 67, 33), (int(pos[0]), int(pos[1])), int(center_size))
            # Add seed pattern
            for i in range(int(center_size * 2)):
                angle_step = i * 137.5  # Golden angle
                radius = math.sqrt(i) * center_size / 6
                seed_x = pos[0] + math.cos(math.radians(angle_step)) * radius
                seed_y = pos[1] - math.sin(math.radians(angle_step)) * radius
                pygame.draw.circle(surface, (60, 40, 20), (int(seed_x), int(seed_y)), 1)
            
            # Draw petals
            for i in range(num_petals):
                petal_angle = angle + (i * (360 / num_petals))
                # Draw long pointed petal
                tip_x = pos[0] + math.cos(math.radians(petal_angle)) * petal_size
                tip_y = pos[1] - math.sin(math.radians(petal_angle)) * petal_size
                left_x = pos[0] + math.cos(math.radians(petal_angle - 15)) * (petal_size * 0.3)
                left_y = pos[1] - math.sin(math.radians(petal_angle - 15)) * (petal_size * 0.3)
                right_x = pos[0] + math.cos(math.radians(petal_angle + 15)) * (petal_size * 0.3)
                right_y = pos[1] - math.sin(math.radians(petal_angle + 15)) * (petal_size * 0.3)
                
                pygame.draw.polygon(surface, flower_color, [
                    (pos[0], pos[1]),
                    (left_x, left_y),
                    (tip_x, tip_y),
                    (right_x, right_y)
                ])

        elif self.plant_type.flower_type == 'lotus':
            # Lotus flower with layered petals
            center_size = size * 0.3
            pygame.draw.circle(surface, (255, 223, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            # Draw multiple layers of petals
            for layer in range(3):
                layer_petals = num_petals - layer * 2
                layer_size = petal_size * (1.0 - layer * 0.2)
                
                for i in range(layer_petals):
                    petal_angle = angle + (i * (360 / layer_petals)) + layer * 15
                    # Create curved petals
                    points = []
                    for t in range(8):
                        t = t / 7
                        radius = layer_size * (0.4 + t * 0.6)
                        curve = math.sin(t * math.pi) * size * 0.25
                        petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                        petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                        petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                        petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                        points.append((petal_x, petal_y))
                    
                    if len(points) >= 3:
                        pygame.draw.polygon(surface, flower_color, points)

        elif self.plant_type.flower_type == 'orchid':
            # Orchid with unique petal arrangement
            center_size = size * 0.15
            pygame.draw.circle(surface, (255, 255, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            # Draw main petal (labellum)
            main_petal_points = []
            main_size = petal_size * 1.2
            for t in range(10):
                t = t / 9
                radius = main_size * (0.3 + t * 0.7)
                curve = math.sin(t * math.pi) * size * 0.4
                petal_x = pos[0] + math.cos(math.radians(angle)) * radius
                petal_y = pos[1] - math.sin(math.radians(angle)) * radius
                petal_x += math.cos(math.radians(angle + 90)) * curve
                petal_y -= math.sin(math.radians(angle + 90)) * curve
                main_petal_points.append((petal_x, petal_y))
            
            if len(main_petal_points) >= 3:
                pygame.draw.polygon(surface, flower_color, main_petal_points)
            
            # Draw side petals
            for i in range(4):
                petal_angle = angle + (i * 72) + 36
                points = []
                for t in range(8):
                    t = t / 7
                    radius = petal_size * (0.3 + t * 0.7)
                    curve = math.sin(t * math.pi) * size * 0.2
                    petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                    petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                    petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                    petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                    points.append((petal_x, petal_y))
                
                if len(points) >= 3:
                    pygame.draw.polygon(surface, flower_color, points)

        elif self.plant_type.flower_type == 'cherry_blossom':
            # Cherry blossom with delicate petals
            center_size = size * 0.2
            pygame.draw.circle(surface, (255, 223, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            for i in range(num_petals):
                petal_angle = angle + (i * (360 / num_petals))
                points = []
                
                # Create heart-shaped petals
                for t in range(12):
                    t = t / 11
                    radius = petal_size * (0.3 + t * 0.7)
                    # Add a slight dip in the middle of each petal
                    radius *= (1 - math.sin(t * math.pi) * 0.1)
                    curve = math.sin(t * math.pi) * size * 0.2
                    petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                    petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                    petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                    petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                    points.append((petal_x, petal_y))
                
                if len(points) >= 3:
                    pygame.draw.polygon(surface, flower_color, points)

        elif self.plant_type.flower_type == 'bird_of_paradise':
            # Bird of Paradise flower
            center_size = size * 0.2
            stem_length = size * 0.8
            
            # Draw the protective bract
            bract_points = [
                (pos[0], pos[1]),
                (pos[0] + math.cos(math.radians(angle + 20)) * stem_length,
                 pos[1] - math.sin(math.radians(angle + 20)) * stem_length),
                (pos[0] + math.cos(math.radians(angle)) * (stem_length * 1.2),
                 pos[1] - math.sin(math.radians(angle)) * (stem_length * 1.2)),
                (pos[0] + math.cos(math.radians(angle - 20)) * stem_length,
                 pos[1] - math.sin(math.radians(angle - 20)) * stem_length),
            ]
            pygame.draw.polygon(surface, (139, 69, 19), bract_points)
            
            # Draw the colorful flower parts
            for i in range(5):
                petal_angle = angle + (i * 15) - 30
                petal_length = size * (0.6 + i * 0.1)
                petal_points = [
                    (pos[0] + math.cos(math.radians(angle)) * size * 0.3,
                     pos[1] - math.sin(math.radians(angle)) * size * 0.3),
                    (pos[0] + math.cos(math.radians(petal_angle)) * petal_length,
                     pos[1] - math.sin(math.radians(petal_angle)) * petal_length),
                    (pos[0] + math.cos(math.radians(petal_angle + 10)) * (petal_length * 0.9),
                     pos[1] - math.sin(math.radians(petal_angle + 10)) * (petal_length * 0.9))
                ]
                pygame.draw.polygon(surface, flower_color, petal_points)

        elif self.plant_type.flower_type == 'cactus_flower':
            # Cactus flower with many delicate petals
            center_size = size * 0.3
            pygame.draw.circle(surface, (255, 223, 0), (int(pos[0]), int(pos[1])), int(center_size))
            
            # Draw multiple layers of thin petals
            for layer in range(3):
                layer_petals = num_petals + layer * 4
                layer_size = petal_size * (0.7 + layer * 0.15)
                
                for i in range(layer_petals):
                    petal_angle = angle + (i * (360 / layer_petals)) + layer * 10
                    points = []
                    
                    # Create thin, pointed petals
                    for t in range(8):
                        t = t / 7
                        radius = layer_size * (0.3 + t * 0.7)
                        # Make petals thinner
                        curve = math.sin(t * math.pi) * size * 0.1
                        petal_x = pos[0] + math.cos(math.radians(petal_angle)) * radius
                        petal_y = pos[1] - math.sin(math.radians(petal_angle)) * radius
                        petal_x += math.cos(math.radians(petal_angle + 90)) * curve
                        petal_y -= math.sin(math.radians(petal_angle + 90)) * curve
                        points.append((petal_x, petal_y))
                    
                    if len(points) >= 3:
                        pygame.draw.polygon(surface, flower_color, points)

    def draw_fractal_leaf(self, surface, pos, angle, size):
        """Draw a fractal-like leaf with veins"""
        if size < 2:
            return
            
        # Adjust leaf droop based on plant state
        if self.state == 'wilting':
            angle -= (1 - self.growth_stage) * 60
            
        # Different leaf shapes based on plant type
        if self.plant_type.leaf_type == 'feather':
            self.draw_feather_leaf(surface, pos, angle, size)
        elif self.plant_type.leaf_type == 'round':
            self.draw_round_leaf(surface, pos, angle, size)
        elif self.plant_type.leaf_type == 'pointed':
            self.draw_pointed_leaf(surface, pos, angle, size)
        elif self.plant_type.leaf_type == 'heart':
            self.draw_heart_leaf(surface, pos, angle, size)
        elif self.plant_type.leaf_type == 'thick':
            self.draw_thick_leaf(surface, pos, angle, size)
        else:
            self.draw_normal_leaf(surface, pos, angle, size)
            
    def draw_normal_leaf(self, surface, pos, angle, size):
        """Original leaf drawing code"""
        # Create leaf curve
        points = []
        ctrl_dist = size * (0.8 if self.state != 'wilting' else 0.5)
        
        # Calculate control points for bezier curve
        left_ctrl = (pos[0] + math.cos(math.radians(angle - 90)) * ctrl_dist,
                    pos[1] - math.sin(math.radians(angle - 90)) * ctrl_dist)
        right_ctrl = (pos[0] + math.cos(math.radians(angle + 90)) * ctrl_dist,
                     pos[1] - math.sin(math.radians(angle + 90)) * ctrl_dist)
        tip = (pos[0] + math.cos(math.radians(angle)) * size,
               pos[1] - math.sin(math.radians(angle)) * size)
        
        # Generate leaf outline
        for t in range(21):
            t = t / 20
            if t <= 0.5:
                # Left side of leaf
                t2 = t * 2
                x = (1-t2)*((1-t2)*pos[0] + t2*left_ctrl[0]) + t2*((1-t2)*left_ctrl[0] + t2*tip[0])
                y = (1-t2)*((1-t2)*pos[1] + t2*left_ctrl[1]) + t2*((1-t2)*left_ctrl[1] + t2*tip[1])
            else:
                # Right side of leaf
                t2 = (t - 0.5) * 2
                x = (1-t2)*((1-t2)*tip[0] + t2*right_ctrl[0]) + t2*((1-t2)*right_ctrl[0] + t2*pos[0])
                y = (1-t2)*((1-t2)*tip[1] + t2*right_ctrl[1]) + t2*((1-t2)*right_ctrl[1] + t2*pos[1])
            points.append((x, y))
            
        if len(points) >= 3:
            pygame.draw.polygon(surface, self.instance_leaf_color, points)
            
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.instance_leaf_color[0] + 20),
                         min(255, self.instance_leaf_color[1] + 20),
                         min(255, self.instance_leaf_color[2] + 20))
            pygame.draw.line(surface, vein_color, pos, tip, max(1, int(size/8)))
            
    def draw_feather_leaf(self, surface, pos, angle, size):
        """Draw a fern-like leaf with multiple small leaflets"""
        # Main stem
        end_pos = (pos[0] + math.cos(math.radians(angle)) * size,
                  pos[1] - math.sin(math.radians(angle)) * size)
        pygame.draw.line(surface, self.instance_leaf_color, pos, end_pos, max(1, int(size/10)))
        
        # Draw small leaflets along the stem
        num_pairs = max(3, int(size / 4))
        for i in range(num_pairs):
            t = i / (num_pairs - 1)
            stem_pos = (pos[0] + (end_pos[0] - pos[0]) * t,
                       pos[1] + (end_pos[1] - pos[1]) * t)
            
            leaflet_size = size * 0.2 * (1 - abs(t - 0.5))  # Smaller at base and tip
            for side in [-1, 1]:
                leaf_angle = angle + 60 * side
                self.draw_normal_leaf(surface, stem_pos, leaf_angle, leaflet_size)
                
    def draw_round_leaf(self, surface, pos, angle, size):
        """Draw a round, bush-like leaf"""
        radius = size * 0.5
        center = (pos[0] + math.cos(math.radians(angle)) * radius,
                 pos[1] - math.sin(math.radians(angle)) * radius)
        
        # Draw filled circle
        pygame.draw.circle(surface, self.instance_leaf_color, (int(center[0]), int(center[1])), int(radius))
        
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.instance_leaf_color[0] + 20),
                         min(255, self.instance_leaf_color[1] + 20),
                         min(255, self.instance_leaf_color[2] + 20))
            for a in range(0, 360, 45):
                end_pos = (center[0] + math.cos(math.radians(a)) * radius * 0.8,
                          center[1] - math.sin(math.radians(a)) * radius * 0.8)
                pygame.draw.line(surface, vein_color, center, end_pos, max(1, int(size/12)))
                
    def draw_pointed_leaf(self, surface, pos, angle, size):
        """Draw a pointed leaf for flowers"""
        # Calculate points for a more pointed leaf
        width = size * 0.3
        tip = (pos[0] + math.cos(math.radians(angle)) * size,
               pos[1] - math.sin(math.radians(angle)) * size)
        left = (pos[0] + math.cos(math.radians(angle - 90)) * width,
                pos[1] - math.sin(math.radians(angle - 90)) * width)
        right = (pos[0] + math.cos(math.radians(angle + 90)) * width,
                 pos[1] - math.sin(math.radians(angle + 90)) * width)
                 
        pygame.draw.polygon(surface, self.instance_leaf_color, [left, tip, right])
        
        # Draw center vein
        if self.state != 'dying':
            vein_color = (min(255, self.instance_leaf_color[0] + 20),
                         min(255, self.instance_leaf_color[1] + 20),
                         min(255, self.instance_leaf_color[2] + 20))
            pygame.draw.line(surface, vein_color, pos, tip, max(1, int(size/10)))
            
    def draw_heart_leaf(self, surface, pos, angle, size):
        """Draw a heart-shaped leaf for vines"""
        # Calculate control points for the heart shape
        tip = (pos[0] + math.cos(math.radians(angle)) * size,
               pos[1] - math.sin(math.radians(angle)) * size)
        
        # Create the two lobes of the heart
        points = []
        for t in range(21):
            t = t / 20
            # Left lobe
            cx = pos[0] + math.cos(math.radians(angle - 45)) * size * 0.7
            cy = pos[1] - math.sin(math.radians(angle - 45)) * size * 0.7
            x = cx + math.cos(math.radians(t * 180 - 90)) * size * 0.4
            y = cy + math.sin(math.radians(t * 180 - 90)) * size * 0.4
            points.append((x, y))
            
        # Right lobe
        for t in range(21):
            t = t / 20
            cx = pos[0] + math.cos(math.radians(angle + 45)) * size * 0.7
            cy = pos[1] - math.sin(math.radians(angle + 45)) * size * 0.7
            x = cx + math.cos(math.radians(t * 180 + 90)) * size * 0.4
            y = cy + math.sin(math.radians(t * 180 + 90)) * size * 0.4
            points.append((x, y))
            
        if len(points) >= 3:
            pygame.draw.polygon(surface, self.instance_leaf_color, points)
            
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.instance_leaf_color[0] + 20),
                         min(255, self.instance_leaf_color[1] + 20),
                         min(255, self.instance_leaf_color[2] + 20))
            pygame.draw.line(surface, vein_color, pos, tip, max(1, int(size/10)))
            
    def draw_thick_leaf(self, surface, pos, angle, size):
        """Draw a thick, succulent-like leaf"""
        # Calculate points for a thick, oval leaf
        width = size * 0.4
        points = []
        num_points = 20
        
        for i in range(num_points + 1):
            t = i / num_points
            # Use sine wave to create oval shape
            curr_width = width * math.sin(t * math.pi)
            length = size * (1 - (1 - t) ** 2)  # Quadratic curve for length
            
            x = pos[0] + math.cos(math.radians(angle)) * length
            y = pos[1] - math.sin(math.radians(angle)) * length
            
            # Add points on both sides of the center line
            left_x = x + math.cos(math.radians(angle - 90)) * curr_width
            left_y = y - math.sin(math.radians(angle - 90)) * curr_width
            points.append((left_x, left_y))
            
        for i in range(num_points, -1, -1):
            t = i / num_points
            curr_width = width * math.sin(t * math.pi)
            length = size * (1 - (1 - t) ** 2)
            
            x = pos[0] + math.cos(math.radians(angle)) * length
            y = pos[1] - math.sin(math.radians(angle)) * length
            
            right_x = x + math.cos(math.radians(angle + 90)) * curr_width
            right_y = y - math.sin(math.radians(angle + 90)) * curr_width
            points.append((right_x, right_y))
            
        if len(points) >= 3:
            pygame.draw.polygon(surface, self.instance_leaf_color, points)
            
        # Draw center line for thickness effect
        if self.state != 'dying':
            highlight_color = (min(255, self.instance_leaf_color[0] + 40),
                             min(255, self.instance_leaf_color[1] + 40),
                             min(255, self.instance_leaf_color[2] + 40))
            end_pos = (pos[0] + math.cos(math.radians(angle)) * size,
                      pos[1] - math.sin(math.radians(angle)) * size)
            pygame.draw.line(surface, highlight_color, pos, end_pos, max(1, int(size/6)))

    def get_nutritional_value(self):
        """Calculate the nutritional value of the plant based on its current state"""
        # Base value depends on plant size and type
        base_value = self.base_size * self.plant_type.mature_height_multiplier
        
        # Multiply by growth stage to account for plant maturity
        maturity_value = base_value * self.growth_stage
        
        # Apply state-based modifiers
        state_multiplier = {
            'seed': 0.3,      # Seeds provide little nutrition
            'sprouting': 0.6,  # Young plants are somewhat nutritious
            'mature': 1.0,     # Mature plants provide full value
            'wilting': 0.5,    # Wilting plants provide less value
            'dying': 0.2      # Dying plants provide very little value
        }.get(self.state, 1.0)
        
        # Plant type specific multipliers (some plants are more nutritious)
        type_multiplier = {
            'fern': 0.8,           # Ferns are less nutritious
            'bush': 1.0,           # Standard nutrition
            'purple_flower': 1.2,   # Flowering plants are more nutritious
            'red_flower': 1.2,
            'yellow_flower': 1.2,
            'vine': 0.9,           # Vines are slightly less nutritious
            'succulent': 1.3,      # Succulents are very nutritious
            'orchid': 1.1          # Orchids are somewhat more nutritious
        }.get(self.plant_type.name, 1.0)
        
        # Calculate raw value
        raw_value = maturity_value * state_multiplier * type_multiplier
        
        # Normalize to range [0.2, 1.0]
        normalized_value = 0.2 + (0.8 * min(1.0, raw_value / (self.base_size * 2)))
        
        return normalized_value

    def calculate_steps_to_reach(self, worm_x, worm_y, worm_speed):
        """Calculate how many time steps it would take for the worm to reach this plant"""
        dx = self.x - worm_x
        dy = self.y - worm_y
        distance = math.sqrt(dx*dx + dy*dy)
        return int(distance / worm_speed)  # Round down to nearest step

    def predict_future_value(self, worm_x, worm_y, worm_speed):
        """Predict the nutritional value when the worm reaches this plant"""
        # Calculate actual time steps needed based on worm's current position and speed
        time_steps = self.calculate_steps_to_reach(worm_x, worm_y, worm_speed)
        
        # Calculate future lifetime when worm would reach the plant
        future_lifetime = max(0, self.lifetime - time_steps)
        future_percentage = future_lifetime / self.max_lifetime
        
        # If the plant will be dead by the time we reach it, return 0
        if future_lifetime <= 0:
            return 0.0
            
        # Get current value for comparison
        current_value = self.get_nutritional_value()
        
        # Calculate growth stage using a continuous function
        # The growth follows this pattern:
        # 1. Seed (90-100%): Very low value, slight increase
        # 2. Sprouting (70-90%): Rapid increase
        # 3. Mature (40-70%): Peak value
        # 4. Wilting (10-40%): Gradual decrease
        # 5. Dying (0-10%): Rapid decrease to minimum
        
        if future_percentage >= 0.9:  # Seed phase
            # Linear increase from 0.1 to 0.2
            growth_stage = 0.1 + (0.1 * (1.0 - future_percentage) / 0.1)
        elif future_percentage >= 0.7:  # Sprouting phase
            # Quadratic increase from 0.2 to 1.0
            phase_progress = (0.9 - future_percentage) / 0.2
            growth_stage = 0.2 + (0.8 * phase_progress * phase_progress)
        elif future_percentage >= 0.4:  # Mature phase
            # Maintain peak value with slight variation
            phase_progress = (0.7 - future_percentage) / 0.3
            growth_stage = 1.0 - (0.1 * (phase_progress - 0.5) * (phase_progress - 0.5))
        elif future_percentage >= 0.1:  # Wilting phase
            # Exponential decrease
            phase_progress = (0.4 - future_percentage) / 0.3
            growth_stage = 0.4 * math.exp(-2 * phase_progress)  # Faster decay
        else:  # Dying phase
            # Linear decrease to 0
            growth_stage = 0.1 * (future_percentage / 0.1)
        
        # Base value calculation
        base_value = self.base_size * self.plant_type.mature_height_multiplier
        maturity_value = base_value * growth_stage
        
        # Plant type multipliers
        type_multiplier = {
            'fern': 0.8,           # Ferns are less nutritious
            'bush': 1.0,           # Standard nutrition
            'purple_flower': 1.2,   # Flowering plants are more nutritious
            'red_flower': 1.2,
            'yellow_flower': 1.2,
            'vine': 0.9,           # Vines are slightly less nutritious
            'succulent': 1.3,      # Succulents are very nutritious
            'orchid': 1.1          # Orchids are somewhat more nutritious
        }.get(self.plant_type.name, 1.0)
        
        # Calculate raw value
        raw_value = maturity_value * type_multiplier
        
        # Normalize to range [0.2, 1.0] just like get_nutritional_value
        normalized_value = 0.2 + (0.8 * min(1.0, raw_value / (self.base_size * 2)))
        
        # During wilting/dying phases, future value cannot exceed current value
        if future_percentage < 0.4:  # If wilting or dying
            normalized_value = min(normalized_value, current_value)
            
        # If almost dead, value approaches zero
        if future_percentage < 0.05:  # Last 5% of lifetime
            # Linear interpolation to zero
            normalized_value *= future_percentage / 0.05
            
        return normalized_value

    def draw(self, surface, worm_x=None, worm_y=None, worm_speed=None, is_target=False):
        """Draw the plant
        
        Args:
            surface: The pygame surface to draw on
            worm_x: The worm's x position (optional)
            worm_y: The worm's y position (optional)
            worm_speed: The worm's speed (optional)
            is_target: Whether this plant is being targeted by the worm (optional)
        """
        if self.state == 'seed':
            self._draw_seed(surface)
            return
            
        # Draw the plant
        self._draw_plant(surface)
        
        # Draw target indicator if this is the current target
        if is_target:
            # Discreet dot selector under the plant, updated to be larger and positioned a bit higher
            dot_radius = 6
            dot_color = (255, 255, 0)  # Yellow
            dot_x = int(self.x)
            dot_y = int(self.y + self.base_size * 0.5)
            pygame.draw.circle(surface, dot_color, (dot_x, dot_y), dot_radius)
        
        # Only show nutritional values if we have worm position info
        if worm_x is not None and worm_y is not None and worm_speed is not None:
            # Current nutritional value
            current_value = self.get_nutritional_value()
            current_text = f"Now: {current_value:.1f}"
            
            # Future nutritional value
            future_value = self.predict_future_value(worm_x, worm_y, worm_speed)
            future_text = f"Future: {future_value:.1f}"
            
            # Font setup
            value_font = pygame.font.Font(None, max(12, int(self.base_size)))
            
            # Helper function to convert value to color
            def value_to_color(value):
                # Green to red gradient based on value
                return (int(255 * (1 - value)), int(255 * value), 0)
            
            # Render current value
            current_color = value_to_color(current_value)
            current_surface = value_font.render(current_text, True, current_color)
            current_rect = current_surface.get_rect()
            current_rect.centerx = self.x
            current_rect.top = self.y + self.base_size
            
            # Render future value
            future_color = value_to_color(future_value)
            future_surface = value_font.render(future_text, True, future_color)
            future_rect = future_surface.get_rect()
            future_rect.centerx = self.x
            future_rect.top = current_rect.bottom
            
            # Draw values
            surface.blit(current_surface, current_rect)
            surface.blit(future_surface, future_rect)

    def _draw_seed(self, surface):
        """Draw the plant in seed state"""
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.base_size)

    def _draw_plant(self, surface):
        """Draw the mature plant"""
        # Draw main stem and branches
        self.draw_fractal_branch(surface, 
                               (self.x, self.y),  # start position
                               90,  # vertical angle
                               self.stem_height,  # length
                               max(1, int(self.base_size/6)),  # width
                               0,  # initial depth
                               self.max_branch_depth)  # max depth

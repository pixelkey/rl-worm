import pygame
import math
import random

class PlantType:
    def __init__(self, name, characteristics):
        self.name = name
        self.base_color = characteristics.get('base_color', (20, 120, 20))
        self.max_branch_depth = characteristics.get('max_branch_depth', 2)
        self.branch_angle_range = characteristics.get('branch_angle_range', (20, 40))
        self.leaf_type = characteristics.get('leaf_type', 'normal')
        self.leaf_size_multiplier = characteristics.get('leaf_size_multiplier', 1.0)
        self.stem_type = characteristics.get('stem_type', 'straight')
        self.growth_speed = characteristics.get('growth_speed', 1.0)
        self.branch_density = characteristics.get('branch_density', (0.6, 1.0))
        self.mature_height_multiplier = characteristics.get('mature_height_multiplier', 1.0)

# Define different plant types
PLANT_TYPES = {
    'fern': PlantType('fern', {
        'base_color': (40, 150, 40),
        'max_branch_depth': 4,
        'branch_angle_range': (30, 60),
        'leaf_type': 'feather',
        'leaf_size_multiplier': 0.7,
        'stem_type': 'curved',
        'growth_speed': 1.2,
        'branch_density': (0.8, 0.9),
        'mature_height_multiplier': 0.8
    }),
    'bush': PlantType('bush', {
        'base_color': (20, 140, 20),
        'max_branch_depth': 3,
        'branch_angle_range': (20, 45),
        'leaf_type': 'round',
        'leaf_size_multiplier': 1.2,
        'stem_type': 'straight',
        'growth_speed': 0.9,
        'branch_density': (0.7, 1.0),
        'mature_height_multiplier': 0.7
    }),
    'flower': PlantType('flower', {
        'base_color': (150, 50, 150),  # Purple base for flower
        'max_branch_depth': 2,
        'branch_angle_range': (15, 30),
        'leaf_type': 'pointed',
        'leaf_size_multiplier': 0.8,
        'stem_type': 'straight',
        'growth_speed': 1.1,
        'branch_density': (0.5, 0.7),
        'mature_height_multiplier': 1.2
    }),
    'vine': PlantType('vine', {
        'base_color': (30, 160, 30),
        'max_branch_depth': 3,
        'branch_angle_range': (40, 80),
        'leaf_type': 'heart',
        'leaf_size_multiplier': 0.9,
        'stem_type': 'wavy',
        'growth_speed': 1.3,
        'branch_density': (0.4, 0.6),
        'mature_height_multiplier': 1.4
    }),
    'succulent': PlantType('succulent', {
        'base_color': (100, 160, 100),
        'max_branch_depth': 2,
        'branch_angle_range': (10, 25),
        'leaf_type': 'thick',
        'leaf_size_multiplier': 1.5,
        'stem_type': 'thick',
        'growth_speed': 0.7,
        'branch_density': (0.9, 1.0),
        'mature_height_multiplier': 0.5
    })
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
        self.color = self.plant_type.base_color
        
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
                'leaf_size_var': random.uniform(0.8, 1.2) if random.random() < self.leaf_density else 1.0
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
            self.growth_stage = (1.0 - life_ratio) * 10  # 0.0 to 1.0 during seed stage
            self.stem_height = int(self.base_size * 0.2)
            self.leaf_size = 0
            self.color = (30, 110, 20)  # Dark green seed
            self.branch_growth = [0.0] * (self.max_branch_depth + 1)
        elif life_ratio > 0.7:  # Next 20% - sprouting
            self.state = 'sprouting'
            stage_progress = (0.9 - life_ratio) / 0.2  # 0.0 to 1.0 during sprouting
            self.growth_stage = stage_progress
            self.stem_height = int(self.base_size * (0.2 + stage_progress * 1.8))
            self.leaf_size = int(self.base_size * stage_progress * 0.5)
            green = int(110 + 70 * stage_progress)
            self.color = (20, green, 10)
            self.branch_growth = [stage_progress] + [0.0] * self.max_branch_depth
        elif life_ratio > 0.4:  # Next 30% - growing/mature
            self.state = 'mature'
            stage_progress = (0.7 - life_ratio) / 0.3  # 0.0 to 1.0 during mature stage
            self.growth_stage = stage_progress
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
            self.growth_stage = stage_progress
            self.branch_growth = [1.0] * (self.max_branch_depth + 1)  # All branches fully grown
            green = int(180 * stage_progress)
            brown = int(120 * (1 - stage_progress))
            self.color = (brown, green, 0)
        else:  # Final 10% - nearly dead
            self.state = 'dying'
            stage_progress = life_ratio / 0.1  # 1.0 to 0.0 during dying
            self.growth_stage = stage_progress
            self.branch_growth = [1.0] * (self.max_branch_depth + 1)  # All branches fully grown
            self.stem_height = int(self.base_size * (1 + stage_progress))
            self.leaf_size = int(self.base_size * stage_progress)
            self.color = (100, int(60 * stage_progress), 0)  # Brown
            
        return self.lifetime > 0

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
                pygame.draw.lines(surface, self.color, False, points, width)
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
                pygame.draw.lines(surface, self.color, False, points, max(1, int(width * growth_progress)))
        
        # Draw leaves using stored pattern, only if branch is grown enough
        if current_pattern['has_leaf'] and self.state not in ['seed', 'sprouting'] and growth_progress > 0.5:
            leaf_size = self.leaf_size * (0.7 ** depth) * current_pattern['leaf_size_var'] * min(1.0, growth_progress * 2 - 1)
            if leaf_size > 0:
                self.draw_fractal_leaf(surface, end, angle, leaf_size)
        
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
            pygame.draw.polygon(surface, self.color, points)
            
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.color[0] + 20),
                         min(255, self.color[1] + 20),
                         min(255, self.color[2] + 20))
            pygame.draw.line(surface, vein_color, pos, tip, max(1, int(size/8)))
            
    def draw_feather_leaf(self, surface, pos, angle, size):
        """Draw a fern-like leaf with multiple small leaflets"""
        # Main stem
        end_pos = (pos[0] + math.cos(math.radians(angle)) * size,
                  pos[1] - math.sin(math.radians(angle)) * size)
        pygame.draw.line(surface, self.color, pos, end_pos, max(1, int(size/10)))
        
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
        pygame.draw.circle(surface, self.color, (int(center[0]), int(center[1])), int(radius))
        
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.color[0] + 20),
                         min(255, self.color[1] + 20),
                         min(255, self.color[2] + 20))
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
                 
        pygame.draw.polygon(surface, self.color, [left, tip, right])
        
        # Draw center vein
        if self.state != 'dying':
            vein_color = (min(255, self.color[0] + 20),
                         min(255, self.color[1] + 20),
                         min(255, self.color[2] + 20))
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
            pygame.draw.polygon(surface, self.color, points)
            
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.color[0] + 20),
                         min(255, self.color[1] + 20),
                         min(255, self.color[2] + 20))
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
            pygame.draw.polygon(surface, self.color, points)
            
        # Draw center line for thickness effect
        if self.state != 'dying':
            highlight_color = (min(255, self.color[0] + 40),
                             min(255, self.color[1] + 40),
                             min(255, self.color[2] + 40))
            end_pos = (pos[0] + math.cos(math.radians(angle)) * size,
                      pos[1] - math.sin(math.radians(angle)) * size)
            pygame.draw.line(surface, highlight_color, pos, end_pos, max(1, int(size/6)))

    def draw(self, surface):
        """Draw the plant"""
        if self.state == 'seed':
            # Draw seed
            radius = max(2, int(self.base_size * 0.3))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), radius)
            return
            
        # Draw main stem and branches
        self.draw_fractal_branch(surface, 
                               (self.x, self.y),  # start position
                               90,  # vertical angle
                               self.stem_height,  # length
                               max(1, int(self.base_size/6)),  # width
                               0,  # initial depth
                               self.max_branch_depth)  # max depth

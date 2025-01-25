import pygame
import math
import random

class Plant:
    def __init__(self, x, y, game_size):
        self.x = x
        self.y = y
        self.base_size = int(game_size/40)  # Base size for scaling
        self.max_lifetime = 800  # Longer lifetime for more visible growth stages
        self.lifetime = self.max_lifetime
        self.state = 'seed'  # seed -> sprouting -> growing -> mature -> wilting
        self.growth_stage = 0.0  # 0.0 to 1.0 for current stage
        self.stem_height = 0
        self.leaf_size = 0
        self.color = (20, 120, 20)  # Start with darker green
        
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
        elif life_ratio > 0.7:  # Next 20% - sprouting
            self.state = 'sprouting'
            self.growth_stage = (0.9 - life_ratio) * 5  # 0.0 to 1.0 during sprouting
            self.stem_height = int(self.base_size * (0.2 + self.growth_stage))
            self.leaf_size = int(self.base_size * self.growth_stage * 0.5)
            green = int(110 + 70 * self.growth_stage)
            self.color = (20, green, 10)
        elif life_ratio > 0.4:  # Next 30% - growing/mature
            self.state = 'mature'
            self.growth_stage = 1.0
            self.stem_height = int(self.base_size * 2)
            self.leaf_size = self.base_size
            self.color = (20, 180, 10)  # Bright healthy green
        elif life_ratio > 0.1:  # Next 30% - starting to wilt
            self.state = 'wilting'
            self.growth_stage = life_ratio / 0.4  # Gradually wilt
            green = int(180 * self.growth_stage)
            brown = int(120 * (1 - self.growth_stage))
            self.color = (brown, green, 0)
        else:  # Final 10% - nearly dead
            self.state = 'dying'
            self.growth_stage = life_ratio / 0.1
            self.stem_height = int(self.base_size * (1 + self.growth_stage))
            self.leaf_size = int(self.base_size * self.growth_stage)
            self.color = (100, int(60 * self.growth_stage), 0)  # Brown
            
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
            
        # Calculate end point using stored variation
        variation = current_pattern['variation']
        end_x = start[0] + math.cos(math.radians(angle)) * length * variation
        end_y = start[1] - math.sin(math.radians(angle)) * length * variation
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
            mid_x = (start[0] + end[0])/2 + ctrl_offset[0]
            mid_y = (start[1] + end[1])/2 + ctrl_offset[1]
            ctrl_point = (mid_x, mid_y)
            
            points = []
            for t in range(10):
                t = t / 9
                # Quadratic bezier for curved branches
                x = (1-t)*((1-t)*start[0] + t*ctrl_point[0]) + t*((1-t)*ctrl_point[0] + t*end[0])
                y = (1-t)*((1-t)*start[1] + t*ctrl_point[1]) + t*((1-t)*ctrl_point[1] + t*end[1])
                points.append((x, y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, self.color, False, points, width)
        
        # Draw leaves using stored pattern
        if current_pattern['has_leaf'] and self.state not in ['seed', 'sprouting']:
            leaf_size = self.leaf_size * (0.7 ** depth) * current_pattern['leaf_size_var']
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
        
        # Create main leaf shape using bezier curve
        points = []
        ctrl_dist = size * (0.8 if self.state != 'wilting' else 0.5)
        
        # Calculate control points for left and right sides of leaf
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
        
        # Draw leaf outline
        if len(points) >= 3:
            pygame.draw.polygon(surface, self.color, points)
        
        # Draw veins
        if self.state != 'dying':
            vein_color = (min(255, self.color[0] + 20),
                         min(255, self.color[1] + 20),
                         min(255, self.color[2] + 20))
            # Main vein
            pygame.draw.line(surface, vein_color, pos, tip, max(1, int(size/8)))
            
            # Side veins
            num_veins = max(2, int(size/4))
            for i in range(num_veins):
                t = (i + 1) / (num_veins + 1)
                start_x = pos[0] + (tip[0] - pos[0]) * t
                start_y = pos[1] + (tip[1] - pos[1]) * t
                
                # Left vein
                end_x = start_x + math.cos(math.radians(angle - 45)) * size * 0.3 * (1 - t)
                end_y = start_y - math.sin(math.radians(angle - 45)) * size * 0.3 * (1 - t)
                pygame.draw.line(surface, vein_color, (start_x, start_y), (end_x, end_y), 1)
                
                # Right vein
                end_x = start_x + math.cos(math.radians(angle + 45)) * size * 0.3 * (1 - t)
                end_y = start_y - math.sin(math.radians(angle + 45)) * size * 0.3 * (1 - t)
                pygame.draw.line(surface, vein_color, (start_x, start_y), (end_x, end_y), 1)
        
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

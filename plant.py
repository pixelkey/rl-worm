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
        self.branch_angles = []  # Store angles for branches
        self.branch_lengths = []  # Store lengths for branches
        
        # Generate random but symmetric branch patterns
        num_branches = random.randint(2, 3)
        for i in range(num_branches):
            angle = random.uniform(20, 50)
            self.branch_angles.extend([angle, -angle])
            length = random.uniform(0.6, 0.9)
            self.branch_lengths.extend([length, length])
        
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
            
        # Calculate end point
        end_x = start[0] + math.cos(math.radians(angle)) * length
        end_y = start[1] - math.sin(math.radians(angle)) * length
        end = (end_x, end_y)
        
        # Draw the branch
        if depth == 0:  # Main stem
            points = []
            num_points = 8
            time_factor = pygame.time.get_ticks() / 1000.0
            wave_offset = math.sin(time_factor * 2) * 2 * self.growth_stage
            
            for i in range(num_points):
                t = i / (num_points - 1)
                wave = math.sin(t * math.pi) * wave_offset
                x = start[0] + t * (end[0] - start[0]) + wave
                y = start[1] + t * (end[1] - start[1])
                points.append((x, y))
            
            if len(points) >= 2:
                pygame.draw.lines(surface, self.color, False, points, width)
        else:  # Sub-branches
            pygame.draw.line(surface, self.color, start, end, max(1, width))
        
        # Draw leaves at the end of branches
        if depth == max_depth and self.state not in ['seed', 'sprouting']:
            self.draw_fractal_leaf(surface, end, angle, self.leaf_size * (0.7 ** depth))
        
        # Calculate new length and width for sub-branches
        new_length = length * 0.7
        new_width = max(1, int(width * 0.7))
        
        # Draw sub-branches
        if self.state not in ['seed', 'sprouting']:
            branch_angles = [30, -30] if depth == 0 else [20, -20]
            for branch_angle in branch_angles:
                new_angle = angle + branch_angle
                if self.state == 'wilting':
                    # Droop the branches when wilting
                    droop = (1 - self.growth_stage) * 30
                    new_angle -= droop
                self.draw_fractal_branch(surface, end, new_angle, new_length, new_width, depth + 1, max_depth)

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
                               2)  # max depth

import pygame
import math
import time

class WormRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.game_surface = pygame.Surface((width, height))
        
        # Colors
        self.background_color = (50, 50, 50)
        self.worm_color = (0, 255, 0)
        self.head_color = (0, 200, 0)
        self.food_color = (255, 0, 0)
        self.eye_color = (255, 255, 255)
        self.pupil_color = (0, 0, 0)
        
    def render_game(self, game_state):
        """Render the game state"""
        self.game_surface.fill(self.background_color)
        
        # Draw food
        for food in game_state['food']:
            pygame.draw.circle(self.game_surface, self.food_color, 
                             (int(food[0]), int(food[1])), 
                             game_state['food_size'])
        
        # Draw worm segments
        for i, pos in enumerate(game_state['positions']):
            color = self.head_color if i == 0 else self.worm_color
            pygame.draw.circle(self.game_surface, color, 
                             (int(pos[0]), int(pos[1])), 
                             game_state['segment_size'])
        
        # Draw face if there are positions
        if game_state['positions']:
            self._draw_face(game_state)
        
        return self.game_surface
    
    def _draw_face(self, game_state):
        """Draw the worm's face"""
        x, y = game_state['positions'][0]
        angle = game_state['angle']
        face_angle = angle - math.pi/2  # Face angle is 90 degrees offset
        segment_size = game_state['segment_size']
        expression = game_state.get('expression', 0)
        
        # Draw eyes
        eye_offset = segment_size * 0.3
        eye_y_offset = segment_size * 0.15
        eye_size = segment_size * 0.25
        pupil_size = eye_size * 0.5
        
        # Calculate eye positions
        base_left_x = -eye_offset
        base_left_y = -eye_y_offset
        base_right_x = eye_offset
        base_right_y = -eye_y_offset
        
        # Rotate eye positions
        left_eye_x = x + (base_left_x * math.cos(face_angle) - base_left_y * math.sin(face_angle))
        left_eye_y = y + (base_left_x * math.sin(face_angle) + base_left_y * math.cos(face_angle))
        right_eye_x = x + (base_right_x * math.cos(face_angle) - base_right_y * math.sin(face_angle))
        right_eye_y = y + (base_right_x * math.sin(face_angle) + base_right_y * math.cos(face_angle))
        
        # Draw eyes (white part)
        pygame.draw.circle(self.game_surface, self.eye_color, 
                         (int(left_eye_x), int(left_eye_y)), int(eye_size))
        pygame.draw.circle(self.game_surface, self.eye_color, 
                         (int(right_eye_x), int(right_eye_y)), int(eye_size))
        
        # Draw pupils
        pygame.draw.circle(self.game_surface, self.pupil_color, 
                         (int(left_eye_x), int(left_eye_y)), int(pupil_size))
        pygame.draw.circle(self.game_surface, self.pupil_color, 
                         (int(right_eye_x), int(right_eye_y)), int(pupil_size))
        
        # Draw mouth
        mouth_width = segment_size * 0.7
        mouth_height = segment_size * 0.3
        mouth_y_offset = segment_size * 0.4
        
        # Base mouth points
        base_left_x = -mouth_width/2
        base_left_y = mouth_y_offset
        base_right_x = mouth_width/2
        base_right_y = mouth_y_offset
        
        # Rotate mouth points
        left_x = x + (base_left_x * math.cos(face_angle) - base_left_y * math.sin(face_angle))
        left_y = y + (base_left_x * math.sin(face_angle) + base_left_y * math.cos(face_angle))
        right_x = x + (base_right_x * math.cos(face_angle) - base_right_y * math.sin(face_angle))
        right_y = y + (base_right_x * math.sin(face_angle) + base_right_y * math.cos(face_angle))
        
        # Calculate control point for curved mouth
        curve_height = mouth_height * expression
        base_control_x = 0
        base_control_y = mouth_y_offset + curve_height
        
        control_x = x + (base_control_x * math.cos(face_angle) - base_control_y * math.sin(face_angle))
        control_y = y + (base_control_x * math.sin(face_angle) + base_control_y * math.cos(face_angle))
        
        # Draw curved mouth
        points = [(int(left_x), int(left_y)), 
                 (int(control_x), int(control_y)),
                 (int(right_x), int(right_y))]
        pygame.draw.lines(self.game_surface, self.pupil_color, False, points, 4)

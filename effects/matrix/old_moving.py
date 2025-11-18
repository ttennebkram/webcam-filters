"""
Matrix Old Moving Effect

Matrix-style falling characters with face-aware flow physics. Each character
behaves like a water particle, detecting edges and faces and flowing around
them with independent lateral movement. Features ultra-dense cascading streams
with physics-based deflection.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class MatrixOldMovingEffect(BaseEffect):
    """Matrix-style falling characters with face-aware flow physics"""

    def __init__(self, width, height):
        super().__init__(width, height)

        # Matrix character set: ASCII only for maximum speed
        self.chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.¦|<>*+-=')

        # Column width for character rain - MAXIMUM streamers with OpenCV speed
        self.col_width = 4  # Ultra narrow = maximum streamers
        self.num_cols = width // self.col_width

        # Initialize rain columns
        self.drops = []
        self.char_spacing = 16  # Smaller vertical spacing for smaller characters

        for i in range(self.num_cols):
            num_chars = 20
            self.drops.append({
                'x': i * self.col_width,
                'y': random.randint(-height, 0),
                'speed': random.uniform(3.29, 13.16),  # 30% faster: 2.53*1.3=3.29, 10.12*1.3=13.16
                'chars': [random.choice(self.chars) for _ in range(num_chars)],
                'char_x_offsets': [0.0] * num_chars,  # Each character's lateral offset (independent)
                'char_x_velocities': [0.0] * num_chars  # Each character's lateral velocity
            })

    @classmethod
    def get_name(cls):
        return "Matrix Old Moving"

    @classmethod
    def get_description(cls):
        return "Matrix rain with physics-based character flow around obstacles"

    @classmethod
    def get_category(cls):
        return "matrix"

    def update(self):
        """Update drop positions - physics handled in update_drops"""
        pass

    def update_drops(self, obstacle_mask=None):
        """Update drop positions with physics - each character independent like water"""
        for drop in self.drops:
            # Update vertical position
            drop['y'] += drop['speed']

            # Reset drop if it falls off screen
            if drop['y'] > self.height:
                drop['y'] = random.randint(-100, -10)
                num_chars = len(drop['chars'])
                drop['chars'] = [random.choice(self.chars) for _ in range(num_chars)]
                drop['char_x_offsets'] = [0.0] * num_chars
                drop['char_x_velocities'] = [0.0] * num_chars
                continue

            # Skip physics if drop is above screen (optimization)
            if drop['y'] < -400:
                continue

            # Update EACH character independently based on what's at its location
            for i in range(len(drop['chars'])):
                char_y = int(drop['y'] - (i * self.char_spacing))

                # Skip characters that are off screen
                if char_y < 0 or char_y >= self.height:
                    continue

                char_x = int(drop['x'] + drop['char_x_offsets'][i])

                # Check if THIS character is hitting an obstacle
                if obstacle_mask is not None and 0 <= char_x < self.width:
                    # Look ahead for collision
                    collision = False
                    is_horizontal_edge = False

                    # Look down from this character's position
                    for dy in range(1, 25):
                        check_y = char_y + dy
                        if check_y >= self.height:
                            break

                        if obstacle_mask[check_y, char_x] > 0:
                            collision = True

                            # Check if horizontal edge - even wider detection
                            h_run = 0
                            for dx in range(-25, 25):  # Even wider range
                                cx = char_x + dx
                                if 0 <= cx < self.width and obstacle_mask[check_y, cx] > 0:
                                    h_run += 1
                            if h_run > 2:  # Very low threshold - catch all angles
                                is_horizontal_edge = True
                            break

                    if collision:
                        # Determine flow direction - search much further
                        left_space = 0
                        right_space = 0

                        for dx in range(1, 90):  # Search much further (was 70)
                            if char_x - dx >= 0 and obstacle_mask[char_y, char_x - dx] == 0:
                                left_space += 1
                            else:
                                break

                        for dx in range(1, 90):  # Search much further (was 70)
                            if char_x + dx < self.width and obstacle_mask[char_y, char_x + dx] == 0:
                                right_space += 1
                            else:
                                break

                        # Apply VERY strong forces for maximum deflection
                        force = 20.0 if is_horizontal_edge else 15.0  # Much stronger!

                        if left_space > right_space:
                            drop['char_x_velocities'][i] -= force
                        else:
                            drop['char_x_velocities'][i] += force
                    else:
                        # Not colliding - dampen lateral movement slower (let it flow more)
                        drop['char_x_velocities'][i] *= 0.85

                # Apply velocity and less damping for more dramatic flow
                drop['char_x_offsets'][i] += drop['char_x_velocities'][i]
                drop['char_x_velocities'][i] *= 0.92  # Less damping for longer flow

                # Wider clamp for more dramatic movement
                drop['char_x_offsets'][i] = max(-250, min(250, drop['char_x_offsets'][i]))

    def draw(self, frame, face_mask=None):
        """Draw the Matrix rain effect using OpenCV for speed"""
        # Update physics with face mask
        self.update_drops(face_mask)

        # Start with a black background
        result = np.zeros_like(frame)

        # Use OpenCV font - much faster than PIL
        font = cv2.FONT_HERSHEY_SIMPLEX

        for drop in self.drops:
            base_x = drop['x']
            y_pos = int(drop['y'])

            # Early exit if entire drop is offscreen (major optimization)
            if y_pos < -400 or y_pos > self.height + 100:
                continue

            # Draw trail of characters - each has its own independent position
            # Only process visible characters for speed
            for i, char in enumerate(drop['chars']):
                char_y = y_pos - (i * self.char_spacing)

                # Skip characters that are offscreen (major speed optimization)
                if char_y < 0 or char_y >= self.height:
                    continue

                # Each character has its own lateral offset (independent movement like water particles)
                char_x_offset = drop['char_x_offsets'][i]
                x_pos = int(base_x + char_x_offset)

                if 0 <= x_pos < self.width:
                    # Only head (i==0) is bold
                    if i == 0:
                        thickness = 2
                    else:
                        thickness = 1

                    # Proper fade from white at head to very dark at tail
                    if i == 0:
                        color = (220, 255, 255)  # BGR: white at head
                    elif i == 1:
                        color = (100, 255, 200)  # BGR: bright green
                    elif i == 2:
                        color = (0, 220, 0)  # BGR: green
                    elif i == 3:
                        color = (0, 180, 0)  # BGR: medium green
                    elif i == 4:
                        color = (0, 140, 0)  # BGR: fading
                    elif i == 5:
                        color = (0, 100, 0)  # BGR: darker
                    elif i < 10:
                        intensity = 100 - ((i - 5) * 15)  # Fade from 100 to 25
                        color = (0, max(25, intensity), 0)
                    else:
                        intensity = 25 - ((i - 10) * 2)  # Fade from 25 to very dark
                        color = (0, max(5, intensity), 0)

                    cv2.putText(result, char, (x_pos, char_y), font, 0.4, color, thickness, cv2.LINE_AA)

        # Brighten background significantly - 50% brightness
        darkened_frame = cv2.convertScaleAbs(frame, alpha=0.50, beta=0)  # 50% brightness

        # Add green tint to background
        green_tint = np.zeros_like(frame)
        green_tint[:, :, 1] = 45  # More green channel for visibility
        background = cv2.addWeighted(darkened_frame, 0.75, green_tint, 0.25, 0)

        # Composite characters on top - characters should be bright
        result = cv2.addWeighted(background, 1.0, result, 1.0, 0)

        return result

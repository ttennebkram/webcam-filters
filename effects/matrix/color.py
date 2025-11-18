"""
Matrix Color Effect

A Matrix-style fixed character grid with brightness-based rendering that
preserves and enhances the original colors of the video feed. Features
cascading streamers with white heads and colored trails that sample colors
from the background.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class MatrixColorEffect(BaseEffect):
    """Matrix-style fixed character grid with brightness-based color rendering"""

    def __init__(self, width, height):
        super().__init__(width, height)

        # Matrix character set: ASCII only for maximum speed
        self.chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.¦|<>*+-=')

        # Character grid settings
        self.char_width = 10  # Horizontal spacing
        self.char_height = 18  # Vertical spacing
        self.num_cols = width // self.char_width
        self.num_rows = height // self.char_height

        # Fixed grid of characters - each position has a character
        self.grid = []
        for row in range(self.num_rows):
            grid_row = []
            for col in range(self.num_cols):
                grid_row.append({
                    'char': random.choice(self.chars),
                    'change_counter': random.randint(0, 30)  # When to change character
                })
            self.grid.append(grid_row)

        # Streamers - waves of illumination moving down columns
        self.streamers = []
        for i in range(self.num_cols):
            if random.random() < 0.3:  # 30% of columns have active streamers
                self.streamers.append({
                    'col': i,
                    'row': random.randint(-20, 0),
                    'speed': random.uniform(0.3, 1.2),  # Rows per frame
                    'length': random.randint(10, 25)  # Length of trail
                })
            else:
                self.streamers.append(None)

    @classmethod
    def get_name(cls):
        return "Matrix Color"

    @classmethod
    def get_description(cls):
        return "Matrix effect with full color preservation and cascading streamers"

    @classmethod
    def get_category(cls):
        return "matrix"

    def update(self):
        """Update character grid and streamers"""
        # Randomly change characters in the grid
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = self.grid[row][col]
                cell['change_counter'] -= 1
                if cell['change_counter'] <= 0:
                    cell['char'] = random.choice(self.chars)
                    cell['change_counter'] = random.randint(20, 40)

        # Update streamers
        for i, streamer in enumerate(self.streamers):
            if streamer is not None:
                # Move streamer down
                streamer['row'] += streamer['speed']

                # Reset if off screen
                if streamer['row'] > self.num_rows + streamer['length']:
                    streamer['row'] = random.randint(-30, -5)
                    streamer['speed'] = random.uniform(0.3, 1.2)
                    streamer['length'] = random.randint(10, 25)
            else:
                # Randomly spawn new streamers
                if random.random() < 0.002:  # Small chance each frame
                    self.streamers[i] = {
                        'col': i,
                        'row': random.randint(-20, 0),
                        'speed': random.uniform(0.3, 1.2),
                        'length': random.randint(10, 25)
                    }

    def draw(self, frame, face_mask=None):
        """Draw the Matrix effect - characters brightness based on background"""
        # Convert to HSV to manipulate saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Boost saturation significantly
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5, beta=0)  # 2.5x saturation

        # Convert back to BGR
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Dim the saturated frame for background
        background = cv2.convertScaleAbs(saturated_frame, alpha=0.3, beta=0)

        # Create edge-detected overlay with color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Soften edges with blur
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Create colored edge overlay by masking saturated frame with edges
        edge_mask = cv2.merge([edges, edges, edges])  # 3-channel mask
        edge_overlay = cv2.bitwise_and(saturated_frame, edge_mask)
        edge_overlay = cv2.convertScaleAbs(edge_overlay, alpha=0.45, beta=0)

        # Combine background with colored edge overlay
        background = cv2.addWeighted(background, 1.0, edge_overlay, 1.0, 0)

        # Start with this background
        result = background.copy()

        # Use OpenCV font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create character layer
        char_layer = np.zeros_like(frame)

        # Draw all characters in the grid
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x_pos = col * self.char_width
                y_pos = row * self.char_height + self.char_height  # Baseline

                if y_pos < 0 or y_pos >= self.height or x_pos >= self.width:
                    continue

                # Sample color and brightness from saturated background at this position
                brightness = gray[min(y_pos, self.height-1), min(x_pos, self.width-1)]
                bg_color = saturated_frame[min(y_pos, self.height-1), min(x_pos, self.width-1)]

                # Check if this position is part of a streamer
                streamer_intensity = 0
                for streamer in self.streamers:
                    if streamer is not None and streamer['col'] == col:
                        # Distance from streamer head (negative = behind, positive = ahead)
                        distance = streamer['row'] - row

                        if distance >= 0 and distance < streamer['length']:
                            if distance == 0:
                                # Head of streamer - white/bright
                                streamer_intensity = 255
                            else:
                                # Tail - fades from bright to dark BEHIND the head
                                fade = 1.0 - (distance / streamer['length'])
                                streamer_intensity = int(200 * fade)

                # Combine background brightness with streamer intensity
                if streamer_intensity > 0:
                    # Streamer - use white for head, colored for tail
                    if streamer_intensity > 240:
                        # Head - white
                        color = (255, 255, 255)
                        thickness = 2
                    else:
                        # Tail - take hue from background, boost saturation
                        intensity_scale = streamer_intensity / 200.0
                        color = tuple(int(c * intensity_scale * 1.5) for c in bg_color)
                        thickness = 1
                else:
                    # No streamer - characters use background color, saturated
                    intensity_scale = brightness / 255.0 * 1.2  # Boost brightness
                    color = tuple(int(c * intensity_scale) for c in bg_color)
                    thickness = 1

                # Check if color is too dark
                if max(color) < 20:
                    continue  # Don't draw very dark characters

                # Get character
                char = self.grid[row][col]['char']

                cv2.putText(char_layer, char, (x_pos, y_pos), font, 0.5, color, thickness, cv2.LINE_AA)

        # Composite characters on top of background
        result = cv2.addWeighted(result, 1.0, char_layer, 1.0, 0)

        return result

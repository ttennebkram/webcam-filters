"""
Color Enhanced Lines Effect

Creates green matrix-like edge visualization with character overlays
based on brightness levels and moving streamers.
"""

import cv2
import numpy as np
import random
from typing import Optional
from core.base_effect import BaseEffect


class LinesColorEnhanced(BaseEffect):
    """Matrix-style edge visualization with character overlays"""

    def __init__(self, width: int, height: int):
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
    def get_name(cls) -> str:
        return "Color Enhanced Matrix"

    @classmethod
    def get_description(cls) -> str:
        return "Green matrix-style edge visualization with animated characters"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

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

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw the Matrix effect - characters brightness based on background"""
        # Create edge-detected background
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Soften edges with blur
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Dim the edges
        edges = cv2.convertScaleAbs(edges, alpha=0.45, beta=0)

        # Convert edges to pure green on black
        edge_background = np.zeros_like(frame)
        edge_background[:, :, 1] = edges  # Green channel only

        # Convert original frame to grayscale, then to green-only
        # This removes all color, keeping only brightness
        dimmed_gray = cv2.convertScaleAbs(gray, alpha=0.15, beta=0)  # Very dim
        dimmed_frame = np.zeros_like(frame)
        dimmed_frame[:, :, 1] = dimmed_gray  # Only green channel - pure green on black

        # Combine dimmed frame with green edges
        background = cv2.addWeighted(dimmed_frame, 1.0, edge_background, 1.0, 0)

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

                # Sample brightness from background at this position
                brightness = gray[min(y_pos, self.height-1), min(x_pos, self.width-1)]

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
                # Streamer overrides background brightness
                if streamer_intensity > 0:
                    final_intensity = streamer_intensity
                else:
                    # No streamer - characters reflect background brightness
                    final_intensity = int(brightness * 0.7)  # Much brighter to see background

                if final_intensity < 20:
                    continue  # Don't draw very dark characters

                # Get character
                char = self.grid[row][col]['char']

                # Color based on intensity and streamer
                if streamer_intensity > 240:
                    # Head - white
                    color = (final_intensity, final_intensity, final_intensity)
                    thickness = 2
                else:
                    # Green
                    color = (0, final_intensity, 0)
                    thickness = 1

                cv2.putText(char_layer, char, (x_pos, y_pos), font, 0.5, color, thickness, cv2.LINE_AA)

        # Composite characters on top of background
        result = cv2.addWeighted(result, 1.0, char_layer, 1.0, 0)

        return result

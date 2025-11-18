"""
Fall effect - Autumn colors with falling leaves.

Applies an autumn color palette mapping (oranges, reds, golds, greens),
with realistic falling leaves that oscillate and rotate as they descend.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class FallEffect(BaseEffect):
    """Autumn scene with fall colors and falling leaves"""

    @classmethod
    def get_name(cls) -> str:
        return "Fall"

    @classmethod
    def get_description(cls) -> str:
        return "Autumn color palette with falling leaves"

    @classmethod
    def get_category(cls) -> str:
        return "seasonal"

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Falling leaves
        self.leaves = []
        for _ in range(15):  # 15 leaves falling at various times
            self.leaves.append({
                'x': random.uniform(0, width),
                'y': random.uniform(-height, 0),  # Start above screen
                'speed': random.uniform(4.5, 10.5),  # Vertical speed (3x faster)
                'oscillation_speed': random.uniform(0.02, 0.05),  # Horizontal oscillation rate
                'oscillation_amplitude': random.uniform(20, 50),  # How far left/right
                'oscillation_phase': random.uniform(0, 2 * np.pi),  # Starting phase
                'rotation': random.uniform(0, 360),
                'rotation_speed': random.uniform(-3, 3),  # Rotation per frame
                'size': random.uniform(15, 30),  # Leaf size
                'color': random.choice([
                    [0, 140, 255],   # Dark Orange
                    [0, 69, 255],    # Orange Red
                    [0, 0, 200],     # Dark Red
                    [0, 215, 255],   # Gold
                    [0, 255, 255],   # Yellow
                    [34, 139, 34],   # Forest Green
                ])
            })

    def update(self):
        """Update falling leaves"""
        for leaf in self.leaves:
            # Move down
            leaf['y'] += leaf['speed']

            # Oscillate left and right (sine wave)
            leaf['oscillation_phase'] += leaf['oscillation_speed']
            leaf['x'] += np.sin(leaf['oscillation_phase']) * leaf['oscillation_amplitude'] * 0.05

            # Rotate
            leaf['rotation'] += leaf['rotation_speed']

            # Reset if off screen
            if leaf['y'] > self.height + 50:
                leaf['y'] = random.uniform(-100, -50)
                leaf['x'] = random.uniform(0, self.width)
                leaf['speed'] = random.uniform(4.5, 10.5)  # 3x faster
                leaf['oscillation_phase'] = random.uniform(0, 2 * np.pi)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Fall leaf effect - map colors to autumn palette"""
        # Downsample to 33% for 9x speed improvement
        original_height, original_width = frame.shape[:2]
        small_frame = cv2.resize(frame, (original_width // 3, original_height // 3), interpolation=cv2.INTER_LINEAR)

        # Define fall leaf color palette (BGR format)
        fall_colors = np.array([
            [0, 100, 0],        # Dark Green
            [34, 139, 34],      # Forest Green
            [50, 205, 50],      # Lime Green
            [0, 128, 128],      # Olive Green
            [47, 107, 85],      # Dark Olive Green
            [0, 140, 255],      # Dark Orange
            [0, 69, 255],       # Orange Red
            [0, 0, 200],        # Dark Red
            [0, 0, 139],        # Maroon
            [0, 165, 255],      # Orange
            [0, 215, 255],      # Gold
            [0, 255, 255],      # Yellow
            [32, 165, 218],     # Golden Rod
            [139, 0, 139],      # Dark Magenta (for purples)
            [128, 0, 128],      # Purple
        ], dtype=np.float32)

        # Convert frame to HSV to identify blue pixels
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Create mask for blue pixels (hue ~90-130 for cyan/blue range)
        blue_mask = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130)).astype(np.uint8)

        # Apply bilateral filter to smooth colors while preserving edges
        smooth = cv2.bilateralFilter(small_frame, 9, 75, 75)

        # For non-blue pixels, map to fall colors
        # Reshape for processing
        pixels = smooth.reshape((-1, 3)).astype(np.float32)

        # Find closest fall color for each pixel
        result_pixels = np.zeros_like(pixels)
        for i in range(len(pixels)):
            # Calculate distance to each fall color
            distances = np.linalg.norm(fall_colors - pixels[i], axis=1)
            # Assign closest fall color
            result_pixels[i] = fall_colors[np.argmin(distances)]

        result = result_pixels.astype(np.uint8).reshape((small_frame.shape))

        # Restore blue pixels from original
        blue_mask_3channel = cv2.merge([blue_mask, blue_mask, blue_mask])
        result = np.where(blue_mask_3channel > 0, small_frame, result)

        # Apply median filter to create smoother, more uniform "glass pieces"
        result = cv2.medianBlur(result, 5)

        # Optional: Add dark edges between segments for leading effect
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Create black lines for leading
        result[edges > 0] = [0, 0, 0]

        # Upscale back to original size using NEAREST to maintain blocky stained glass look
        result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Draw falling leaves
        for leaf in self.leaves:
            x = int(leaf['x'])
            y = int(leaf['y'])

            if -50 <= x < self.width + 50 and -50 <= y < self.height + 50:
                # Create a realistic leaf shape with pointed tip
                size = int(leaf['size'])

                # Create leaf shape points - teardrop/pointed leaf shape
                leaf_points = []

                # Top pointed tip
                leaf_points.append((0, -size * 0.8))

                # Right side - curved
                leaf_points.append((size * 0.3, -size * 0.4))
                leaf_points.append((size * 0.5, 0))
                leaf_points.append((size * 0.3, size * 0.4))

                # Bottom rounded
                leaf_points.append((0, size * 0.5))

                # Left side - curved
                leaf_points.append((-size * 0.3, size * 0.4))
                leaf_points.append((-size * 0.5, 0))
                leaf_points.append((-size * 0.3, -size * 0.4))

                # Rotate leaf
                rot_rad = np.radians(leaf['rotation'])
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)

                rotated_points = []
                for px, py in leaf_points:
                    rx = px * cos_r - py * sin_r
                    ry = px * sin_r + py * cos_r
                    rotated_points.append((int(x + rx), int(y + ry)))

                # Draw filled polygon for leaf
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], tuple(int(c) for c in leaf['color']), cv2.LINE_AA)

                # Add stem (small line from center)
                stem_angle = rot_rad + np.pi / 2
                stem_length = size * 0.3
                stem_end_x = int(x + stem_length * np.cos(stem_angle))
                stem_end_y = int(y + stem_length * np.sin(stem_angle))
                cv2.line(result, (x, y), (stem_end_x, stem_end_y), (0, 100, 0), 2, cv2.LINE_AA)

        return result

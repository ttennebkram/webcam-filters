"""
Winter effect - Arctic blue coloring with white snowflakes.

Applies arctic blue tinting near edges that fades to white in center areas,
with falling white snowflakes.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class WinterEffect(BaseEffect):
    """Winter scene with arctic blue tones and snowflakes"""

    @classmethod
    def get_name(cls) -> str:
        return "Winter"

    @classmethod
    def get_description(cls) -> str:
        return "Arctic blue coloring with falling white snowflakes"

    @classmethod
    def get_category(cls) -> str:
        return "seasonal"

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Snowflakes - irregular shapes
        self.snowflakes = []
        for _ in range(300):  # Heavy snow
            # Create small irregular shape for each snowflake
            num_points = random.randint(3, 5)
            base_size = random.uniform(1.5, 4.0)
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * np.pi
                radius = base_size * random.uniform(0.6, 1.0)
                px = radius * np.cos(angle)
                py = radius * np.sin(angle)
                points.append((px, py))

            self.snowflakes.append({
                'x': random.uniform(0, width),
                'y': random.uniform(-height, 0),  # Start above screen
                'speed': random.uniform(2.0, 6.0),
                'points': points,  # Irregular shape
                'drift': random.uniform(-0.5, 0.5),  # Horizontal drift
                'rotation': random.uniform(0, 360)
            })

    def update(self):
        """Update snowflakes"""
        for flake in self.snowflakes:
            # Move down
            flake['y'] += flake['speed']
            flake['x'] += flake['drift']
            flake['rotation'] += random.uniform(-2, 2)  # Slight rotation

            # Reset if off screen
            if flake['y'] > self.height:
                flake['y'] = random.uniform(-20, 0)
                flake['x'] = random.uniform(0, self.width)
                flake['speed'] = random.uniform(2.0, 6.0)
                flake['drift'] = random.uniform(-0.5, 0.5)

                # Regenerate irregular shape
                num_points = random.randint(3, 5)
                base_size = random.uniform(1.5, 4.0)
                points = []
                for i in range(num_points):
                    angle = (i / num_points) * 2 * np.pi
                    radius = base_size * random.uniform(0.6, 1.0)
                    px = radius * np.cos(angle)
                    py = radius * np.sin(angle)
                    points.append((px, py))
                flake['points'] = points

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Winter effect - arctic blue near edges, white in center"""
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Use only Gaussian blurs for completely smooth, non-blocky edges
        # Start with the raw edges
        edges_processed = edges.copy().astype(np.float32)

        # Progressive blur cycles - no dilation to avoid blockiness
        # Each cycle extends reach smoothly with increasingly large blurs for fuzzier edges
        blur_sizes = [
            (15, 15), (21, 21), (27, 27), (33, 33), (39, 39), (45, 45),
            (51, 51), (57, 57), (63, 63), (71, 71), (81, 81), (91, 91)
        ]

        for blur_size in blur_sizes:
            edges_processed = cv2.GaussianBlur(edges_processed, blur_size, 0)
            # Boost opacity slightly after each blur to maintain strength
            edges_processed = np.clip(edges_processed * 1.15, 0, 255)

        # Add additional blur to feather edges more, then boost center back to 100%
        edges_blurred = cv2.GaussianBlur(edges_processed, (151, 151), 0)
        # Boost to restore center brightness while keeping feathered edges
        edges_blurred = np.clip(edges_blurred * 1.3, 0, 255).astype(np.uint8)

        # Convert frame to HSV to preserve brightness while changing color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Invert the mask - high values where there are NO edges, low values near edges
        inverted_mask = 255 - edges_blurred
        inverted_mask_normalized = inverted_mask.astype(np.float32) / 255.0

        # Apply power curve to extend arctic blue further - higher power = more arctic blue
        inverted_mask_normalized = np.power(inverted_mask_normalized, 5.0)  # 5th power for much more arctic blue

        # Near edges (inverted_mask low): arctic blue (hue 105)
        # Away from edges (inverted_mask high): white (saturation 0)

        # Blend hue: arctic blue everywhere
        hsv[:, :, 0] = 105  # Arctic blue hue

        # Blend saturation: high near edges (arctic blue), low away from edges (white)
        hsv[:, :, 1] = 255 * (1.0 - inverted_mask_normalized)  # Low saturation = white, max saturation near edges

        # Keep original brightness (value channel)
        # hsv[:, :, 2] stays unchanged

        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Now add white glow right at the edges on top
        # Use the original edge mask with less blur for tighter white edges
        white_edges = cv2.GaussianBlur(edges_processed, (31, 31), 0)
        white_edges = np.clip(white_edges * 1.2, 0, 255).astype(np.uint8)

        # Convert to 3-channel for blending
        white_edges_3channel = cv2.merge([white_edges, white_edges, white_edges])

        # Create white color
        white = np.ones_like(result, dtype=np.uint8) * 255

        # Blend white on top of arctic blue at edges
        alpha = white_edges_3channel.astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - alpha) + white.astype(np.float32) * alpha).astype(np.uint8)

        # Draw snowflakes - irregular white blobs
        for flake in self.snowflakes:
            x = int(flake['x'])
            y = int(flake['y'])

            if -10 <= x < self.width + 10 and -10 <= y < self.height + 10:
                # Rotate and position the irregular shape points
                rot_rad = np.radians(flake['rotation'])
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)

                rotated_points = []
                for px, py in flake['points']:
                    # Rotate
                    rx = px * cos_r - py * sin_r
                    ry = px * sin_r + py * cos_r
                    # Translate to position
                    rotated_points.append((int(x + rx), int(y + ry)))

                # Draw filled polygon for irregular snowflake
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], (255, 255, 255), cv2.LINE_AA)

        return result

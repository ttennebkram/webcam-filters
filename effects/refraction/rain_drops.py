"""
Rain Drops Refraction Effect

Animated water drops that refract the image as they drip down the screen,
creating a realistic rain or melting effect.
"""

import cv2
import numpy as np
import random
from typing import Optional
from core.base_effect import BaseEffect


class RefractionRainDrops(BaseEffect):
    """Water drops refract the image as they drip down"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Pre-calculate refraction maps for different drop sizes
        self.drop_sizes = [8, 10, 12, 15, 18]  # Smaller drop widths
        self.refraction_maps = {}

        for size in self.drop_sizes:
            self.refraction_maps[size] = self.create_refraction_map(size)

        # Active drops - each drop has position, size, and speed
        self.drops = []

        # Spawn initial drops
        for _ in range(1000):
            self.spawn_drop()

    @classmethod
    def get_name(cls) -> str:
        return "Rain Drops"

    @classmethod
    def get_description(cls) -> str:
        return "Animated water drops that refract the image as they fall"

    @classmethod
    def get_category(cls) -> str:
        return "refraction"

    def create_refraction_map(self, drop_width):
        """Pre-calculate refraction displacement map for a water drop shape"""
        drop_height = int(drop_width * 2.0)  # Drops are elongated
        half_w = drop_width // 2

        # Create displacement maps and alpha mask
        offset_x = np.zeros((drop_height, drop_width), dtype=np.float32)
        offset_y = np.zeros((drop_height, drop_width), dtype=np.float32)
        alpha = np.zeros((drop_height, drop_width), dtype=np.float32)

        # Create water drop shape with refraction effect
        for y in range(drop_height):
            for x in range(drop_width):
                # Distance from center x-axis
                dx = x - half_w
                fx = x - half_w + 0.5  # Sub-pixel center

                # Normalize y position (0 at top, 1 at bottom)
                ny = y / drop_height

                # Drop shape: stretched tail at top, rounded bulb at bottom
                if ny < 0.5:
                    # Top half - narrow stretched tail
                    max_radius = half_w * (0.2 + 0.5 * (ny / 0.5))
                    # Tail has lower alpha - lets things through
                    alpha_strength = 0.3 + 0.5 * (ny / 0.5)
                elif ny < 0.85:
                    # Middle - widening to rounded bottom
                    progress = (ny - 0.5) / 0.35
                    # Use sine curve for smooth rounding
                    max_radius = half_w * (0.7 + 0.3 * np.sin(progress * np.pi / 2))
                    alpha_strength = 0.8 + 0.2 * progress
                else:
                    # Bottom tip - smooth rounded point using cosine
                    progress = (ny - 0.85) / 0.15
                    # Smooth curve to a point
                    max_radius = half_w * np.cos(progress * np.pi / 2)
                    alpha_strength = 1.0

                # Use floating point distance for smoother edges
                dist_from_center = np.sqrt(fx * fx)

                if dist_from_center < max_radius:
                    # Inside the drop - apply refraction
                    # Refraction strength based on distance from edge
                    edge_dist = max_radius - dist_from_center
                    strength = edge_dist / max_radius

                    # Strong refraction at edges (like real water drops)
                    # Center has minimal distortion, edges have maximum
                    if strength < 0.3:
                        # Near edge - very strong refraction with smooth gradient
                        refract_strength = 60.0 * (1.0 - strength / 0.3)
                    else:
                        # Center - moderate refraction
                        refract_strength = 15.0

                    # Smooth falloff at edges for anti-aliasing - wider feathering
                    edge_falloff = min(1.0, edge_dist / 4.0)  # Wider feather

                    offset_x[y, x] = np.sign(dx) * refract_strength
                    offset_y[y, x] = refract_strength * 0.3
                    alpha[y, x] = edge_falloff * alpha_strength

        return {
            'width': drop_width,
            'height': drop_height,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'alpha': alpha
        }

    def spawn_drop(self):
        """Create a new drop at random position"""
        drop = {
            'x': random.randint(0, self.width - 1),
            'y': random.randint(-100, 0),  # Start above screen
            'size': random.choice(self.drop_sizes),
            'speed': random.uniform(8.0, 24.0)  # Drop speed
        }
        self.drops.append(drop)

    def update(self):
        """Update drop positions"""
        # Move drops down
        for drop in self.drops:
            drop['y'] += drop['speed']

        # Remove drops that are off screen and spawn new ones
        self.drops = [d for d in self.drops if d['y'] < self.height + 50]

        # Maintain drop count
        while len(self.drops) < 1000:
            self.spawn_drop()

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw water drops refracting the background image"""
        # Boost saturation and contrast of source frame for more visible drops
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.3, beta=0)  # 1.3x saturation
        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.15, beta=-10)  # Subtle contrast boost
        enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Start with original frame (not enhanced)
        result = frame.copy()

        # Sort drops by y position (top to bottom) so overlapping drops blend correctly
        sorted_drops = sorted(self.drops, key=lambda d: d['y'])

        # Apply each drop's refraction with alpha blending
        for drop in sorted_drops:
            drop_x = int(drop['x'])
            drop_y = int(drop['y'])
            size = drop['size']

            # Get pre-calculated refraction map
            refraction = self.refraction_maps[size]
            drop_width = refraction['width']
            drop_height = refraction['height']
            offset_x = refraction['offset_x']
            offset_y = refraction['offset_y']
            alpha = refraction['alpha']

            # Calculate drop bounds on screen
            left = drop_x - drop_width // 2
            right = left + drop_width
            top = drop_y
            bottom = drop_y + drop_height

            # Skip if completely off screen
            if right < 0 or left >= self.width or bottom < 0 or top >= self.height:
                continue

            # Clip to screen bounds
            screen_left = max(0, left)
            screen_right = min(self.width, right)
            screen_top = max(0, top)
            screen_bottom = min(self.height, bottom)

            # Map to drop coordinates
            drop_left_offset = screen_left - left
            drop_right_offset = drop_left_offset + (screen_right - screen_left)
            drop_top_offset = screen_top - top
            drop_bottom_offset = drop_top_offset + (screen_bottom - screen_top)

            # Extract the region from refraction maps
            region_offset_x = offset_x[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]
            region_offset_y = offset_y[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]
            region_alpha = alpha[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]

            # Create coordinate grids for this region
            region_h = screen_bottom - screen_top
            region_w = screen_right - screen_left

            y_coords, x_coords = np.mgrid[screen_top:screen_bottom, screen_left:screen_right]

            # Calculate source coordinates with refraction
            source_x = (x_coords + region_offset_x.astype(np.int32)).clip(0, self.width - 1)
            source_y = (y_coords + region_offset_y.astype(np.int32)).clip(0, self.height - 1)

            # Get refracted pixels from ENHANCED frame (saturated/contrasted)
            refracted = enhanced_frame[source_y, source_x]

            # Get original pixels
            original = result[screen_top:screen_bottom, screen_left:screen_right]

            # Alpha blend (vectorized)
            alpha_3d = region_alpha[:, :, np.newaxis]  # Add channel dimension
            blended = (refracted * alpha_3d + original * (1.0 - alpha_3d)).astype(np.uint8)

            # Only update pixels with significant alpha
            mask = region_alpha > 0.01
            result[screen_top:screen_bottom, screen_left:screen_right][mask] = blended[mask]

        return result

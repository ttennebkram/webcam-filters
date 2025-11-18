"""
Summer effect - Golden sunset with heat wave refraction.

Applies a warm golden/sunset color temperature with thermal heat wave
refraction effects that rise from bottom to top, creating a shimmering
heat distortion.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class SummerEffect(BaseEffect):
    """Summer scene with golden sunset and heat wave refraction"""

    @classmethod
    def get_name(cls) -> str:
        return "Summer"

    @classmethod
    def get_description(cls) -> str:
        return "Golden sunset with thermal heat wave refraction"

    @classmethod
    def get_category(cls) -> str:
        return "seasonal"

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Heat waves - horizontal cylinders with refraction
        self.heat_waves = []
        for _ in range(8):  # 8 heat waves at various positions
            # Create slightly irregular horizontal cylinder
            wave_width = random.randint(width // 3, width // 2)
            wave_height = random.randint(40, 80)

            # Precompute refraction displacement map for this heat wave
            # Create a lens-like distortion pattern
            displacement_map = np.zeros((wave_height, wave_width, 2), dtype=np.float32)

            for y in range(wave_height):
                for x in range(wave_width):
                    # Distance from center of wave (normalized)
                    center_y = wave_height / 2.0
                    dist_from_center = abs(y - center_y) / center_y

                    # Create refraction effect - strongest at center, weak at edges
                    # Use sine wave for smooth falloff
                    strength = (1.0 - dist_from_center) * np.sin(dist_from_center * np.pi)

                    # Horizontal displacement (wave-like pattern)
                    x_offset = strength * np.sin(x * 0.1 + y * 0.05) * 8.0

                    # Vertical displacement (slight upward distortion)
                    y_offset = strength * 3.0

                    displacement_map[y, x] = [x_offset, y_offset]

            # Blur the displacement map for smoother refraction
            displacement_map[:, :, 0] = cv2.GaussianBlur(displacement_map[:, :, 0], (15, 15), 0)
            displacement_map[:, :, 1] = cv2.GaussianBlur(displacement_map[:, :, 1], (15, 15), 0)

            self.heat_waves.append({
                'x': random.randint(0, width - wave_width),
                'y': random.randint(height, height + 200),  # Start below screen
                'speed': random.uniform(0.8, 2.0),  # Upward speed
                'width': wave_width,
                'height': wave_height,
                'displacement_map': displacement_map,
                'opacity': random.uniform(0.6, 0.9)
            })

    def update(self):
        """Update heat waves"""
        for wave in self.heat_waves:
            # Move upward
            wave['y'] -= wave['speed']

            # Reset if off screen (disappeared near middle or above)
            if wave['y'] < -wave['height']:
                # Restart from bottom
                wave['y'] = random.randint(self.height, self.height + 200)
                wave['x'] = random.randint(0, self.width - wave['width'])
                wave['speed'] = random.uniform(0.8, 2.0)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Golden sunset effect with thermal heat waves - replace hue with golden sunlight while preserving S and V"""
        # Apply heat wave refraction first
        result = frame.copy()

        for wave in self.heat_waves:
            wave_x = int(wave['x'])
            wave_y = int(wave['y'])
            wave_w = wave['width']
            wave_h = wave['height']

            # Only process if wave is visible on screen
            if wave_y < self.height and wave_y + wave_h > 0:
                # Calculate visible portion of wave
                src_y_start = max(0, -wave_y)
                src_y_end = min(wave_h, self.height - wave_y)
                dst_y_start = max(0, wave_y)
                dst_y_end = min(self.height, wave_y + wave_h)

                src_x_start = max(0, -wave_x)
                src_x_end = min(wave_w, self.width - wave_x)
                dst_x_start = max(0, wave_x)
                dst_x_end = min(self.width, wave_x + wave_w)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    # Get the region to refract
                    region = result[dst_y_start:dst_y_end, dst_x_start:dst_x_end].copy()
                    region_h, region_w = region.shape[:2]

                    # Get corresponding displacement map section
                    disp_map = wave['displacement_map'][src_y_start:src_y_end, src_x_start:src_x_end]

                    # Apply refraction using displacement map
                    for y in range(region_h):
                        for x in range(region_w):
                            if y < disp_map.shape[0] and x < disp_map.shape[1]:
                                dx, dy = disp_map[y, x]

                                # Calculate source pixel with displacement
                                src_x = int(x + dx)
                                src_y = int(y + dy)

                                # Clamp to region bounds
                                src_x = max(0, min(region_w - 1, src_x))
                                src_y = max(0, min(region_h - 1, src_y))

                                # Copy refracted pixel (blend with opacity)
                                if 0 <= src_y < region_h and 0 <= src_x < region_w:
                                    result[dst_y_start + y, dst_x_start + x] = region[src_y, src_x]

        # Convert to HSV to manipulate hue
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Golden sunset hue in OpenCV (0-179 range)
        # Golden/orange sunset is around 15-25 in HSV
        # Using 20 for warm golden sunlight
        golden_hue = 20

        # Replace all hues with golden sunset hue, keep original S and V
        hsv[:, :, 0] = golden_hue

        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Apply gradient of golden sunlight - gentle gradient down to halfway
        height, width = result.shape[:2]

        # Create gradient mask - gentler, extends to halfway down
        gradient = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            # Smooth gradient from 0.5 at top to 0.0 at halfway down
            if y < height // 2:
                # Linear fade from 0.5 to 0.0
                gradient[y, :] = 0.5 * (1.0 - (y / (height // 2)))
            else:
                gradient[y, :] = 0.0

        # Convert gradient to 3-channel
        gradient_3channel = cv2.merge([gradient, gradient, gradient])

        # Create bright golden sunlight color
        bright_golden = np.ones_like(result, dtype=np.uint8)
        bright_golden[:, :] = [100, 200, 255]  # Very bright golden/yellow

        # Blend sunlight gradient on top of result
        result = (result.astype(np.float32) * (1.0 - gradient_3channel) +
                  bright_golden.astype(np.float32) * gradient_3channel).astype(np.uint8)

        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Blur edges for soft effect
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges, edges, edges])

        # Create golden color (BGR format)
        # Golden/orange sunset color
        golden = np.ones_like(result, dtype=np.uint8)
        golden[:, :] = [0, 165, 255]  # BGR: bright golden orange

        # Blend golden edges on top of golden frame
        alpha = edges_3channel.astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - alpha) + golden.astype(np.float32) * alpha).astype(np.uint8)

        return result

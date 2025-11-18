"""
Spring effect - Light sky blue with hunter green edges and Easter eggs.

Applies a gentle sky blue gradient from the top, hunter green edge highlighting,
grass blades at the bottom, and colorful Easter eggs nestled in the grass.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class SpringEffect(BaseEffect):
    """Spring scene with sky blue, grass, and Easter eggs"""

    @classmethod
    def get_name(cls) -> str:
        return "Spring"

    @classmethod
    def get_description(cls) -> str:
        return "Light sky blue with hunter green edges, grass, and Easter eggs"

    @classmethod
    def get_category(cls) -> str:
        return "seasonal"

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Generate grass blades along bottom
        self.grass_blades = []
        grass_height_max = 50  # Maximum grass height (shorter)
        num_blades = 200  # Dense grass

        for _ in range(num_blades):
            x = random.randint(0, width)
            # Grass height varies
            height_offset = random.randint(25, grass_height_max)
            # More varied angles for natural look
            angle = random.uniform(-45, 45)
            # Width variation
            blade_width = random.randint(2, 4)
            # Color variation - darker hunter green base
            color_variance = random.randint(-15, 15)
            base_color = np.array([20, 60, 20])  # Darker hunter green
            color = np.clip(base_color + color_variance, 0, 255)

            self.grass_blades.append({
                'x': x,
                'height': height_offset,
                'angle': angle,
                'width': blade_width,
                'color': tuple(map(int, color))
            })

        # Generate Easter eggs
        self.easter_eggs = []
        num_eggs = 8

        for _ in range(num_eggs):
            # Position near bottom, in grass area
            x = random.randint(50, width - 50)
            y = height - random.randint(20, 50)  # Nestled in grass

            # Size variation - 50% larger
            egg_width = random.randint(30, 45)
            egg_height = int(egg_width * 1.4)  # Eggs are taller than wide

            # Random orientation angle
            rotation_angle = random.randint(0, 360)

            # Egg colors - pastel spring colors
            color_choices = [
                [180, 150, 255],  # Pastel pink (BGR)
                [255, 200, 150],  # Pastel blue
                [150, 220, 255],  # Pastel yellow
                [180, 255, 200],  # Pastel green
                [220, 180, 220],  # Pastel purple
                [200, 220, 255],  # Pale peach
            ]
            base_color = random.choice(color_choices)

            # Pattern type
            pattern = random.choice(['dots', 'stripes', 'solid'])

            self.easter_eggs.append({
                'x': x,
                'y': y,
                'width': egg_width,
                'height': egg_height,
                'color': base_color,
                'pattern': pattern,
                'rotation': rotation_angle
            })

    def update(self):
        """Update animation"""
        pass

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Spring effect with light sky blue gradient and hunter green edges"""
        result = frame.copy()

        # Apply gradient of light sky blue - gentle gradient down to halfway
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

        # Create light sky blue color
        light_sky_blue = np.ones_like(result, dtype=np.uint8)
        light_sky_blue[:, :] = [235, 206, 135]  # BGR: light sky blue

        # Blend sky blue gradient on top of result
        result = (result.astype(np.float32) * (1.0 - gradient_3channel) +
                  light_sky_blue.astype(np.float32) * gradient_3channel).astype(np.uint8)

        # Detect edges - lower thresholds to keep more edges stable
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)  # Lower thresholds for more stable edges

        # Dilate edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Blur edges more at the edges for softer falloff
        edges_blurred = cv2.GaussianBlur(edges_dilated, (15, 15), 0)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges_blurred, edges_blurred, edges_blurred])

        # Create hunter green color (BGR format)
        hunter_green = np.ones_like(result, dtype=np.uint8)
        hunter_green[:, :] = [35, 86, 35]  # BGR: hunter green

        # Blend hunter green edges on top of frame with stronger alpha for more visible green
        alpha = edges_3channel.astype(np.float32) / 255.0 * 1.5  # Boost alpha for more green
        alpha = np.clip(alpha, 0, 1.0)  # Clamp to valid range
        result = (result.astype(np.float32) * (1.0 - alpha) + hunter_green.astype(np.float32) * alpha).astype(np.uint8)

        # Draw Easter eggs first (so grass can overlap them)
        for egg in self.easter_eggs:
            cx, cy = egg['x'], egg['y']
            w, h = egg['width'], egg['height']
            rotation = egg['rotation']

            # Create egg shape (ellipse) with rotation
            overlay = result.copy()

            # Main egg color with subtle shading (rotated ellipse)
            cv2.ellipse(overlay, (cx, cy), (w // 2, h // 2), rotation, 0, 360, egg['color'], -1)

            # Add subtle highlight for dimension (rotated offset)
            highlight_angle = np.radians(rotation - 45)
            highlight_offset_x = int(w // 6 * np.cos(highlight_angle))
            highlight_offset_y = int(h // 6 * np.sin(highlight_angle))
            highlight_color = [min(255, c + 40) for c in egg['color']]
            cv2.ellipse(overlay, (cx - highlight_offset_x, cy - highlight_offset_y),
                       (w // 4, h // 4), rotation, 0, 360, highlight_color, -1)

            # Add subtle shadow (bottom-right, accounting for rotation)
            shadow_angle = np.radians(rotation + 135)
            shadow_offset_x = int(w // 8 * np.cos(shadow_angle))
            shadow_offset_y = int(h // 8 * np.sin(shadow_angle))
            shadow_color = [max(0, c - 30) for c in egg['color']]
            cv2.ellipse(overlay, (cx - shadow_offset_x, cy - shadow_offset_y),
                       (w // 3, h // 3), rotation, 0, 360, shadow_color, -1)

            # Add pattern
            if egg['pattern'] == 'dots':
                # Small dots
                for _ in range(random.randint(6, 10)):
                    dot_x = cx + random.randint(-w // 3, w // 3)
                    dot_y = cy + random.randint(-h // 3, h // 3)
                    dot_color = [max(0, c - 40) for c in egg['color']]
                    cv2.circle(overlay, (dot_x, dot_y), 3, dot_color, -1)

            elif egg['pattern'] == 'stripes':
                # Stripes perpendicular to egg orientation
                stripe_color = [max(0, c - 40) for c in egg['color']]
                for i in range(3):
                    stripe_offset = -h // 3 + i * h // 3
                    # Calculate stripe position along egg's major axis
                    stripe_x = int(cx + stripe_offset * np.sin(np.radians(rotation)))
                    stripe_y = int(cy - stripe_offset * np.cos(np.radians(rotation)))
                    cv2.ellipse(overlay, (stripe_x, stripe_y), (w // 2, 4),
                               rotation + 90, 0, 360, stripe_color, -1)

            # Blend egg with soft edges
            alpha_egg = 0.85
            result = cv2.addWeighted(overlay, alpha_egg, result, 1 - alpha_egg, 0)

        # Draw grass blades
        for blade in self.grass_blades:
            x = blade['x']
            base_y = self.height
            tip_y = self.height - blade['height']

            # Calculate slight curve and angle
            angle_rad = np.radians(blade['angle'])
            tip_x = int(x + blade['height'] * np.sin(angle_rad) * 0.3)

            # Draw blade as a thin tapered line
            # Create points for a tapered blade shape
            thickness = blade['width']

            # Draw main blade
            cv2.line(result, (x, base_y), (tip_x, tip_y), blade['color'], thickness)

            # Add slight highlight on one side for depth
            highlight_color = tuple(min(255, c + 20) for c in blade['color'])
            cv2.line(result, (x, base_y), (tip_x, tip_y), highlight_color, max(1, thickness - 1))

        return result

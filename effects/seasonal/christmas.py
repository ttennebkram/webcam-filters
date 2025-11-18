"""
Christmas effect - Festive winter scene with pine garlands, ornaments, and colorful snowflakes.

Applies warm tungsten lighting, gold-edged highlights, pine garland borders
with ornament decorations, and colored snowflakes.
"""

import cv2
import numpy as np
import random
from core.base_effect import BaseEffect


class ChristmasEffect(BaseEffect):
    """Christmas decorations with warm lighting and festive elements"""

    @classmethod
    def get_name(cls) -> str:
        return "Christmas"

    @classmethod
    def get_description(cls) -> str:
        return "Festive decorations with pine garlands, ornaments, and colored snowflakes"

    @classmethod
    def get_category(cls) -> str:
        return "seasonal"

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Christmas light colors - fully saturated
        self.christmas_colors = [
            (0, 0, 255),      # Pure red
            (0, 255, 0),      # Pure green
            (255, 0, 0),      # Pure blue
            (0, 215, 255),    # Pure gold/yellow
            (255, 0, 255),    # Pure magenta
            (0, 255, 255),    # Pure yellow
            (255, 100, 0),    # Pure cyan/light blue
            (128, 0, 255),    # Pure orange-red
        ]

        # Ornament ball colors - reflective appearance
        ornament_colors = [
            (0, 0, 200),      # Deep red
            (0, 180, 0),      # Deep green
            (0, 180, 220),    # Deep gold
        ]

        # Pre-generate ornament balls with fixed positions and colors
        self.ornament_balls = []
        garland_depth = 50

        # Top border balls
        for i in range(0, width, 80):
            if i > 40 and i < width - 40:
                wave_offset = int(10 * np.sin(i * 0.05))
                self.ornament_balls.append({
                    'x': i,
                    'y': garland_depth + wave_offset - 5,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Bottom border balls (offset)
        for i in range(40, width, 80):
            if i > 40 and i < width - 40:
                wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
                self.ornament_balls.append({
                    'x': i,
                    'y': height - garland_depth + wave_offset + 5,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Left border balls
        for i in range(0, height, 80):
            if i > 40 and i < height - 40:
                wave_offset = int(10 * np.sin(i * 0.05))
                self.ornament_balls.append({
                    'x': garland_depth + wave_offset - 5,
                    'y': i,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Right border balls (offset)
        for i in range(40, height, 80):
            if i > 40 and i < height - 40:
                wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
                self.ornament_balls.append({
                    'x': width - garland_depth + wave_offset + 5,
                    'y': i,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Snowflakes - irregular shapes with christmas colors
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
                'rotation': random.uniform(0, 360),
                'color': random.choice(self.christmas_colors)  # Random christmas color
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

                # Regenerate irregular shape and new color
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
                flake['color'] = random.choice(self.christmas_colors)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Christmas effect - warm glow with saturated gold edges"""
        # Apply warm tungsten bulb color temperature
        # Convert to LAB color space for better color temperature control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Shift colors toward warm tungsten (yellow-orange) - toned down
        # A channel: green-red, B channel: blue-yellow
        lab[:, :, 1] = lab[:, :, 1] + 5   # Shift slightly toward red
        lab[:, :, 2] = lab[:, :, 2] + 15  # Shift moderately toward yellow/orange

        # Clamp values
        lab = np.clip(lab, 0, 255)

        # Convert back to BGR
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to make them bolder
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Blur edges for soft effect
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Boost edge intensity to make them more prominent
        edges = cv2.convertScaleAbs(edges, alpha=1.5, beta=0)
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges, edges, edges])

        # Create very bright saturated GOLD for edges
        saturated_gold = np.ones_like(result, dtype=np.uint8)
        saturated_gold[:, :] = [0, 215, 255]  # Saturated gold (BGR)

        # Blend saturated gold edges on top of frame with stronger alpha
        alpha = (edges_3channel.astype(np.float32) / 255.0) * 1.3
        alpha = np.clip(alpha, 0, 1)
        result = (result.astype(np.float32) * (1.0 - alpha) + saturated_gold.astype(np.float32) * alpha).astype(np.uint8)

        # Draw pine garland border around frame - solid bushy appearance
        garland_depth = 50
        hunter_green_dark = (15, 70, 35)  # Dark hunter green
        hunter_green = (20, 85, 45)  # Medium hunter green
        hunter_green_light = (25, 95, 55)  # Light hunter green

        # Create dense pine garland on all four sides
        for i in range(0, self.width, 3):
            # Top border
            wave_offset = int(10 * np.sin(i * 0.05))
            depth = garland_depth + wave_offset

            # Draw overlapping foliage for dense appearance
            for offset in range(-2, 3):
                y_pos = depth + offset * 3
                # Random needle clusters
                for _ in range(3):
                    needle_x = i + random.randint(-4, 4)
                    needle_y = y_pos + random.randint(-6, 6)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle - np.pi/2))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle - np.pi/2))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

            # Bottom border
            wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
            depth = self.height - garland_depth + wave_offset

            for offset in range(-2, 3):
                y_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = i + random.randint(-4, 4)
                    needle_y = y_pos + random.randint(-6, 6)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle + np.pi/2))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle + np.pi/2))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

        for i in range(0, self.height, 3):
            # Left border
            wave_offset = int(10 * np.sin(i * 0.05))
            depth = garland_depth + wave_offset

            for offset in range(-2, 3):
                x_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = x_pos + random.randint(-6, 6)
                    needle_y = i + random.randint(-4, 4)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle - np.pi))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle - np.pi))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

            # Right border
            wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
            depth = self.width - garland_depth + wave_offset

            for offset in range(-2, 3):
                x_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = x_pos + random.randint(-6, 6)
                    needle_y = i + random.randint(-4, 4)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

        # Draw ornament balls with reflective appearance
        for ball in self.ornament_balls:
            center = (ball['x'], ball['y'])
            radius = ball['size']
            base_color = ball['color']

            # Draw main ball with gradient effect for reflection
            # Create multiple circles with decreasing intensity for smooth gradient
            for i in range(radius, 0, -1):
                # Calculate color interpolation from bright center to darker edges
                t = i / radius
                # Darker base color
                color = tuple(int(c * (0.6 + 0.4 * t)) for c in base_color)
                cv2.circle(result, center, i, color, -1, cv2.LINE_AA)

            # Add bright highlight spot (upper left) for glossy reflection
            highlight_offset_x = int(-radius * 0.35)
            highlight_offset_y = int(-radius * 0.35)
            highlight_center = (ball['x'] + highlight_offset_x, ball['y'] + highlight_offset_y)
            highlight_radius = max(2, int(radius * 0.4))

            # Bright version of the color for highlight
            highlight_color = tuple(min(255, int(c * 1.8)) for c in base_color)
            cv2.circle(result, highlight_center, highlight_radius, highlight_color, -1, cv2.LINE_AA)

            # Add small white specular highlight for extra shine
            specular_center = (ball['x'] + int(-radius * 0.25), ball['y'] + int(-radius * 0.25))
            specular_radius = max(1, int(radius * 0.2))
            cv2.circle(result, specular_center, specular_radius, (255, 255, 255), -1, cv2.LINE_AA)

            # Add subtle darker outline for depth
            outline_color = tuple(int(c * 0.4) for c in base_color)
            cv2.circle(result, center, radius, outline_color, 1, cv2.LINE_AA)

        # Draw snowflakes - irregular colored christmas lights
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

                # Draw filled polygon with christmas light color
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], flake['color'], cv2.LINE_AA)

        return result

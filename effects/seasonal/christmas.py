"""
Christmas effect - Festive winter scene with pine garlands, ornaments, and colorful snowflakes.

Applies warm tungsten lighting, gold-edged highlights, pine garland borders
with ornament decorations, and colored snowflakes.
"""

import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ChristmasEffect(BaseUIEffect):
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

    def __init__(self, width: int, height: int, root=None):
        super().__init__(width, height, root)

        # Default values
        self.defaults = {
            'enable_garland': True,
            'enable_ornaments': True,
            'enable_snowflakes': True,
            'enable_warm_lighting': True,
            'enable_gold_edges': True,
            'edge_threshold': 150,
            'gold_hue': 30,
            'gold_saturation': 255,
            'gold_value': 255,
            'num_ornaments': 40,
        }

        # UI control variables
        self.enable_garland = tk.BooleanVar(value=self.defaults['enable_garland'])
        self.enable_ornaments = tk.BooleanVar(value=self.defaults['enable_ornaments'])
        self.enable_snowflakes = tk.BooleanVar(value=self.defaults['enable_snowflakes'])
        self.enable_warm_lighting = tk.BooleanVar(value=self.defaults['enable_warm_lighting'])
        self.enable_gold_edges = tk.BooleanVar(value=self.defaults['enable_gold_edges'])

        # Edge detection sensitivity (upper threshold)
        self.edge_threshold = tk.IntVar(value=self.defaults['edge_threshold'])

        # Gold color HSV - split into separate controls
        self.gold_hue = tk.IntVar(value=self.defaults['gold_hue'])  # 0-179 for OpenCV HSV
        self.gold_saturation = tk.IntVar(value=self.defaults['gold_saturation'])  # 0-255
        self.gold_value = tk.IntVar(value=self.defaults['gold_value'])  # 0-255

        # Number of ornament balls (total across all edges)
        self.num_ornaments = tk.IntVar(value=self.defaults['num_ornaments'])

        # Snow type: None, White, or Colored
        self.snow_type = tk.StringVar(value='Colored')
        self.defaults['snow_type'] = 'Colored'

        # Snow size: Small, Medium, or Large
        self.snow_size = tk.StringVar(value='Small')
        self.defaults['snow_size'] = 'Small'

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
        self.ornament_colors = [
            (0, 0, 200),      # Deep red
            (0, 180, 0),      # Deep green
            (0, 180, 220),    # Deep gold
        ]

        # Pre-generate ornament balls with fixed positions and colors
        self.ornament_balls = []
        self._generate_ornaments()

        # Snowflakes - irregular shapes with christmas colors
        self.snowflakes = []
        self._generate_snowflakes()

    def _generate_ornaments(self):
        """Generate ornament balls along garland borders"""
        self.ornament_balls = []
        garland_depth = 50

        num_balls = self.num_ornaments.get()
        if num_balls <= 0:
            return

        # Calculate perimeter and spacing
        perimeter = 2 * (self.width + self.height)
        spacing = perimeter / num_balls

        # Distribute balls evenly around all four edges
        current_pos = 0
        for _ in range(num_balls):
            # Determine which edge and position
            if current_pos < self.width:
                # Top edge
                x = int(current_pos)
                if 40 < x < self.width - 40:
                    wave_offset = int(10 * np.sin(x * 0.05))
                    self.ornament_balls.append({
                        'x': x,
                        'y': garland_depth + wave_offset - 5,
                        'size': random.randint(12, 18),
                        'color': random.choice(self.ornament_colors)
                    })
            elif current_pos < self.width + self.height:
                # Right edge
                y = int(current_pos - self.width)
                if 40 < y < self.height - 40:
                    wave_offset = int(10 * np.sin(y * 0.05 + np.pi))
                    self.ornament_balls.append({
                        'x': self.width - garland_depth + wave_offset + 5,
                        'y': y,
                        'size': random.randint(12, 18),
                        'color': random.choice(self.ornament_colors)
                    })
            elif current_pos < 2 * self.width + self.height:
                # Bottom edge (going right to left)
                x = int(self.width - (current_pos - self.width - self.height))
                if 40 < x < self.width - 40:
                    wave_offset = int(10 * np.sin(x * 0.05 + np.pi))
                    self.ornament_balls.append({
                        'x': x,
                        'y': self.height - garland_depth + wave_offset + 5,
                        'size': random.randint(12, 18),
                        'color': random.choice(self.ornament_colors)
                    })
            else:
                # Left edge (going bottom to top)
                y = int(self.height - (current_pos - 2 * self.width - self.height))
                if 40 < y < self.height - 40:
                    wave_offset = int(10 * np.sin(y * 0.05))
                    self.ornament_balls.append({
                        'x': garland_depth + wave_offset - 5,
                        'y': y,
                        'size': random.randint(12, 18),
                        'color': random.choice(self.ornament_colors)
                    })

            current_pos += spacing

    def _generate_snowflakes(self):
        """Generate snowflakes with random colors"""
        self.snowflakes = []

        # Determine size range based on snow_size setting
        size_mode = self.snow_size.get()
        if size_mode == 'Small':
            base_size_range = (1.5, 4.0)  # Current small size
        elif size_mode == 'Medium':
            base_size_range = (3.0, 7.0)
        else:  # 'Large'
            base_size_range = (5.0, 10.0)

        for _ in range(300):  # Heavy snow
            # Create irregular shape for each snowflake
            num_points = random.randint(3, 5)
            base_size = random.uniform(*base_size_range)
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * np.pi
                radius = base_size * random.uniform(0.6, 1.0)
                px = radius * np.cos(angle)
                py = radius * np.sin(angle)
                points.append((px, py))

            self.snowflakes.append({
                'x': random.uniform(0, self.width),
                'y': random.uniform(-self.height, 0),  # Start above screen
                'speed': random.uniform(2.0, 6.0),
                'points': points,  # Irregular shape
                'drift': random.uniform(-0.5, 0.5),  # Horizontal drift
                'rotation': random.uniform(0, 360),
                'color': random.choice(self.christmas_colors)  # Random christmas color
            })

    def create_control_panel(self, parent):
        """Create Tkinter control panel"""
        panel = ttk.Frame(parent, padding=10)

        # Enable/Disable toggles
        ttk.Label(panel, text="Elements", font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 5))

        ttk.Checkbutton(panel, text="Pine Garland Border", variable=self.enable_garland).pack(anchor='w', pady=2)
        ttk.Checkbutton(panel, text="Ornament Balls", variable=self.enable_ornaments).pack(anchor='w', pady=2)
        ttk.Checkbutton(panel, text="Warm Tungsten Lighting", variable=self.enable_warm_lighting).pack(anchor='w', pady=2)

        # Number of ornaments slider
        ttk.Label(panel, text="Number of Ornaments (Total):").pack(anchor='w', pady=(5, 0))
        ornament_frame = ttk.Frame(panel)
        ornament_frame.pack(fill='x', pady=2)
        ornament_slider = ttk.Scale(ornament_frame, from_=0, to=100, variable=self.num_ornaments, orient='horizontal',
                                    command=lambda v: self.num_ornaments.set(int(float(v))))
        ornament_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(ornament_frame, textvariable=self.num_ornaments, width=5).pack(side='left', padx=(5, 0))

        # Trace changes to regenerate ornaments
        self.num_ornaments.trace_add('write', lambda *args: self._generate_ornaments())

        # Trace changes to regenerate snowflakes when size changes
        self.snow_size.trace_add('write', lambda *args: self._generate_snowflakes())

        # Snow type and size dropdowns
        snow_frame = ttk.Frame(panel)
        snow_frame.pack(fill='x', pady=(5, 2))

        ttk.Label(snow_frame, text="Snowflakes:").pack(side='left', padx=(0, 5))
        snow_combo = ttk.Combobox(snow_frame, textvariable=self.snow_type, state='readonly', width=10)
        snow_combo['values'] = ('None', 'White', 'Colored')
        snow_combo.pack(side='left', padx=(0, 10))

        ttk.Label(snow_frame, text="Size:").pack(side='left', padx=(0, 5))
        size_combo = ttk.Combobox(snow_frame, textvariable=self.snow_size, state='readonly', width=8)
        size_combo['values'] = ('Small', 'Medium', 'Large')
        size_combo.pack(side='left')

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        # Edge Detection with Gold Edge Glow checkbox
        ttk.Label(panel, text="Edge Detection", font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 5))

        # Gold Edge Glow checkbox
        ttk.Checkbutton(panel, text="Gold Edge Glow", variable=self.enable_gold_edges).pack(anchor='w', pady=2)

        # Edge Sensitivity slider
        ttk.Label(panel, text="Edge Sensitivity (Upper Threshold):").pack(anchor='w', pady=(5, 0))
        threshold_frame = ttk.Frame(panel)
        threshold_frame.pack(fill='x', pady=2)
        ttk.Scale(threshold_frame, from_=50, to=300, variable=self.edge_threshold, orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(threshold_frame, textvariable=self.edge_threshold, width=5).pack(side='left', padx=(5, 0))

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        # Gold Color Controls
        ttk.Label(panel, text="Gold Edge Color (HSV)", font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 5))

        # Hue
        ttk.Label(panel, text="Hue (0-179):").pack(anchor='w', pady=(5, 0))
        hue_frame = ttk.Frame(panel)
        hue_frame.pack(fill='x', pady=2)
        ttk.Scale(hue_frame, from_=0, to=179, variable=self.gold_hue, orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(hue_frame, textvariable=self.gold_hue, width=5).pack(side='left', padx=(5, 0))

        # Saturation
        ttk.Label(panel, text="Saturation (0-255):").pack(anchor='w', pady=(5, 0))
        sat_frame = ttk.Frame(panel)
        sat_frame.pack(fill='x', pady=2)
        ttk.Scale(sat_frame, from_=0, to=255, variable=self.gold_saturation, orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(sat_frame, textvariable=self.gold_saturation, width=5).pack(side='left', padx=(5, 0))

        # Value
        ttk.Label(panel, text="Brightness (0-255):").pack(anchor='w', pady=(5, 0))
        val_frame = ttk.Frame(panel)
        val_frame.pack(fill='x', pady=2)
        ttk.Scale(val_frame, from_=0, to=255, variable=self.gold_value, orient='horizontal').pack(side='left', fill='x', expand=True)
        ttk.Label(val_frame, textvariable=self.gold_value, width=5).pack(side='left', padx=(5, 0))

        # Color preview
        self.color_preview_canvas = tk.Canvas(panel, width=100, height=30, bg='gray')
        self.color_preview_canvas.pack(pady=5)

        # Update preview when values change
        def update_preview(*args):
            h, s, v = self.gold_hue.get(), self.gold_saturation.get(), self.gold_value.get()
            # Convert HSV to RGB for display
            hsv_color = np.uint8([[[h, s, v]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            rgb_color = (int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0]))
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
            self.color_preview_canvas.configure(bg=hex_color)

        self.gold_hue.trace_add('write', update_preview)
        self.gold_saturation.trace_add('write', update_preview)
        self.gold_value.trace_add('write', update_preview)
        update_preview()  # Initial preview

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=10)

        # Buttons
        button_frame = ttk.Frame(panel)
        button_frame.pack(fill='x', pady=5)

        ttk.Button(button_frame, text="Restore Defaults", command=self._restore_defaults).pack(side='left', padx=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self._save_settings).pack(side='left')

        return panel

    def _restore_defaults(self):
        """Restore all settings to default values"""
        self.enable_garland.set(self.defaults['enable_garland'])
        self.enable_ornaments.set(self.defaults['enable_ornaments'])
        self.enable_snowflakes.set(self.defaults['enable_snowflakes'])
        self.enable_warm_lighting.set(self.defaults['enable_warm_lighting'])
        self.enable_gold_edges.set(self.defaults['enable_gold_edges'])
        self.edge_threshold.set(self.defaults['edge_threshold'])
        self.gold_hue.set(self.defaults['gold_hue'])
        self.gold_saturation.set(self.defaults['gold_saturation'])
        self.gold_value.set(self.defaults['gold_value'])
        self.num_ornaments.set(self.defaults['num_ornaments'])
        self.snow_type.set(self.defaults['snow_type'])
        self.snow_size.set(self.defaults['snow_size'])

    def _save_settings(self):
        """Save current settings (placeholder - could save to file)"""
        print("Settings saved!")
        print(f"  Garland: {self.enable_garland.get()}")
        print(f"  Ornaments: {self.enable_ornaments.get()} (count: {self.num_ornaments.get()})")
        print(f"  Snow: {self.snow_type.get()}")
        print(f"  Warm Lighting: {self.enable_warm_lighting.get()}")
        print(f"  Gold Edges: {self.enable_gold_edges.get()}")
        print(f"  Edge Threshold: {self.edge_threshold.get()}")
        print(f"  Gold HSV: ({self.gold_hue.get()}, {self.gold_saturation.get()}, {self.gold_value.get()})")
        # TODO: Could save to JSON file in user's home directory

    def update(self):
        """Update snowflakes"""
        if self.snow_type.get() == 'None':
            return

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

                # Regenerate irregular shape and new color based on current size setting
                size_mode = self.snow_size.get()
                if size_mode == 'Small':
                    base_size_range = (1.5, 4.0)
                elif size_mode == 'Medium':
                    base_size_range = (3.0, 7.0)
                else:  # 'Large'
                    base_size_range = (5.0, 10.0)

                num_points = random.randint(3, 5)
                base_size = random.uniform(*base_size_range)
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
        result = frame.copy()

        # Apply warm tungsten bulb color temperature
        if self.enable_warm_lighting.get():
            # Convert to LAB color space for better color temperature control
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Shift colors toward warm tungsten (yellow-orange) - toned down
            # A channel: green-red, B channel: blue-yellow
            lab[:, :, 1] = lab[:, :, 1] + 5   # Shift slightly toward red
            lab[:, :, 2] = lab[:, :, 2] + 15  # Shift moderately toward yellow/orange

            # Clamp values
            lab = np.clip(lab, 0, 255)

            # Convert back to BGR
            result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Apply gold edges
        if self.enable_gold_edges.get():
            # Detect edges
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, self.edge_threshold.get())

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

            # Create saturated gold from HSV values
            h, s, v = self.gold_hue.get(), self.gold_saturation.get(), self.gold_value.get()
            hsv_color = np.uint8([[[h, s, v]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

            saturated_gold = np.ones_like(result, dtype=np.uint8)
            saturated_gold[:, :] = bgr_color

            # Blend saturated gold edges on top of frame with stronger alpha
            alpha = (edges_3channel.astype(np.float32) / 255.0) * 1.3
            alpha = np.clip(alpha, 0, 1)
            result = (result.astype(np.float32) * (1.0 - alpha) + saturated_gold.astype(np.float32) * alpha).astype(np.uint8)

        # Draw pine garland border around frame
        if self.enable_garland.get():
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
        if self.enable_ornaments.get():
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

        # Draw snowflakes based on type
        snow_mode = self.snow_type.get()
        if snow_mode != 'None':
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

                    # Determine color based on snow type
                    if snow_mode == 'White':
                        snow_color = (255, 255, 255)  # Pure white
                    else:  # 'Colored'
                        snow_color = flake['color']  # Christmas colors

                    # Draw filled polygon
                    pts = np.array(rotated_points, dtype=np.int32)
                    cv2.fillPoly(result, [pts], snow_color, cv2.LINE_AA)

        return result

"""
Hough Circle detection effect using OpenCV.

Detects circles in images using the Hough Transform.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class HoughCirclesEffect(BaseUIEffect):
    """Detect and draw circles using Hough Transform"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_circles = None

        # HoughCircles parameters
        self.dp = tk.DoubleVar(value=1.0)  # Inverse ratio of accumulator resolution
        self.min_dist = tk.IntVar(value=50)  # Min distance between circle centers
        self.param1 = tk.IntVar(value=100)  # Canny high threshold
        self.param2 = tk.IntVar(value=30)  # Accumulator threshold
        self.min_radius = tk.IntVar(value=10)
        self.max_radius = tk.IntVar(value=100)

        # Drawing options
        self.circle_color_b = tk.IntVar(value=0)
        self.circle_color_g = tk.IntVar(value=255)
        self.circle_color_r = tk.IntVar(value=0)
        self.thickness = tk.IntVar(value=2)
        self.draw_center = tk.BooleanVar(value=True)

    @classmethod
    def get_name(cls) -> str:
        return "Hough Circles"

    @classmethod
    def get_description(cls) -> str:
        return "Detect circles using Hough Transform"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def create_control_panel(self, parent):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section
        header_frame = ttk.Frame(self.control_panel)
        header_frame.pack(fill='x', **padding)

        # Title
        title_label = ttk.Label(
            header_frame,
            text="Hough Circles",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.HoughCircles(image, method, dp, minDist, ...)",
            font=('TkFixedFont', 12)
        )
        signature_label.pack(anchor='w', pady=(2, 2))

        # Main frame with two columns
        main_frame = ttk.Frame(self.control_panel)
        main_frame.pack(fill='x', **padding)

        # Left column - Enabled checkbox
        left_column = ttk.Frame(main_frame)
        left_column.pack(side='left', fill='y', padx=(0, 15))

        ttk.Frame(left_column).pack(expand=True)
        enabled_cb = ttk.Checkbutton(
            left_column,
            text="Enabled",
            variable=self.enabled
        )
        enabled_cb.pack()
        ttk.Frame(left_column).pack(expand=True)

        # Right column - all controls
        right_column = ttk.Frame(main_frame)
        right_column.pack(side='left', fill='both', expand=True)

        # Input option
        input_frame = ttk.Frame(right_column)
        input_frame.pack(fill='x', pady=3)

        ttk.Label(input_frame, text="Input:").pack(side='left')

        ttk.Radiobutton(
            input_frame,
            text="Original Image",
            variable=self.show_original,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            input_frame,
            text="Black",
            variable=self.show_original,
            value=False
        ).pack(side='left', padx=5)

        # Output option
        output_frame = ttk.Frame(right_column)
        output_frame.pack(fill='x', pady=3)

        ttk.Label(output_frame, text="Output:").pack(side='left')

        ttk.Radiobutton(
            output_frame,
            text="Draw Features",
            variable=self.draw_features,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            output_frame,
            text="Raw Values Only",
            variable=self.draw_features,
            value=False
        ).pack(side='left', padx=5)

        # Min distance between circles
        mindist_frame = ttk.Frame(right_column)
        mindist_frame.pack(fill='x', pady=3)

        ttk.Label(mindist_frame, text="Min Distance:").pack(side='left')

        mindist_slider = ttk.Scale(
            mindist_frame,
            from_=1,
            to=200,
            orient='horizontal',
            variable=self.min_dist,
            command=self._on_mindist_change
        )
        mindist_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.mindist_label = ttk.Label(mindist_frame, text="50")
        self.mindist_label.pack(side='left', padx=5)

        # Param1 (Canny high threshold)
        param1_frame = ttk.Frame(right_column)
        param1_frame.pack(fill='x', pady=3)

        ttk.Label(param1_frame, text="Param1 (Canny):").pack(side='left')

        param1_slider = ttk.Scale(
            param1_frame,
            from_=1,
            to=300,
            orient='horizontal',
            variable=self.param1,
            command=self._on_param1_change
        )
        param1_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.param1_label = ttk.Label(param1_frame, text="100")
        self.param1_label.pack(side='left', padx=5)

        # Param2 (accumulator threshold)
        param2_frame = ttk.Frame(right_column)
        param2_frame.pack(fill='x', pady=3)

        ttk.Label(param2_frame, text="Param2 (Accum):").pack(side='left')

        param2_slider = ttk.Scale(
            param2_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.param2,
            command=self._on_param2_change
        )
        param2_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.param2_label = ttk.Label(param2_frame, text="30")
        self.param2_label.pack(side='left', padx=5)

        # Min radius
        minrad_frame = ttk.Frame(right_column)
        minrad_frame.pack(fill='x', pady=3)

        ttk.Label(minrad_frame, text="Min Radius:").pack(side='left')

        minrad_slider = ttk.Scale(
            minrad_frame,
            from_=0,
            to=200,
            orient='horizontal',
            variable=self.min_radius,
            command=self._on_minrad_change
        )
        minrad_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.minrad_label = ttk.Label(minrad_frame, text="10")
        self.minrad_label.pack(side='left', padx=5)

        # Max radius
        maxrad_frame = ttk.Frame(right_column)
        maxrad_frame.pack(fill='x', pady=3)

        ttk.Label(maxrad_frame, text="Max Radius:").pack(side='left')

        maxrad_slider = ttk.Scale(
            maxrad_frame,
            from_=0,
            to=500,
            orient='horizontal',
            variable=self.max_radius,
            command=self._on_maxrad_change
        )
        maxrad_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.maxrad_label = ttk.Label(maxrad_frame, text="100")
        self.maxrad_label.pack(side='left', padx=5)

        # Circle color
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Circle Color (BGR):").pack(side='left')

        ttk.Label(color_frame, text="B:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.circle_color_b).pack(side='left')

        ttk.Label(color_frame, text="G:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.circle_color_g).pack(side='left')

        ttk.Label(color_frame, text="R:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.circle_color_r).pack(side='left')

        # Thickness and center dot
        options_frame = ttk.Frame(right_column)
        options_frame.pack(fill='x', pady=3)

        ttk.Label(options_frame, text="Thickness:").pack(side='left')

        thick_slider = ttk.Scale(
            options_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.thickness,
            command=self._on_thick_change
        )
        thick_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thick_label = ttk.Label(options_frame, text="2")
        self.thick_label.pack(side='left', padx=5)

        # Draw center checkbox
        center_frame = ttk.Frame(right_column)
        center_frame.pack(fill='x', pady=3)

        ttk.Checkbutton(
            center_frame,
            text="Draw Center Point",
            variable=self.draw_center
        ).pack(side='left')

        # Circles found display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Circles found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_mindist_change(self, value):
        self.mindist_label.config(text=str(int(float(value))))

    def _on_param1_change(self, value):
        self.param1_label.config(text=str(int(float(value))))

    def _on_param2_change(self, value):
        self.param2_label.config(text=str(int(float(value))))

    def _on_minrad_change(self, value):
        self.minrad_label.config(text=str(int(float(value))))

    def _on_maxrad_change(self, value):
        self.maxrad_label.config(text=str(int(float(value))))

    def _on_thick_change(self, value):
        self.thick_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw circles on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        dp = self.dp.get()
        min_dist = self.min_dist.get()
        param1 = self.param1.get()
        param2 = self.param2.get()
        min_radius = self.min_radius.get()
        max_radius = self.max_radius.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.circle_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.circle_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.circle_color_r.get()
        except Exception:
            color_r = 0
        color = (color_b, color_g, color_r)

        thickness = self.thickness.get()

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        # Store raw values for pipeline use
        self.detected_circles = circles

        circles_count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles_count = len(circles[0])

            if self.draw_features.get():
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]

                    # Draw circle
                    cv2.circle(result, center, radius, color, thickness)

                    # Draw center point
                    if self.draw_center.get():
                        cv2.circle(result, center, 2, color, 3)

        # Update count
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(circles_count))

        return result

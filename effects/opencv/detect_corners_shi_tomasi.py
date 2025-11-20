"""
Shi-Tomasi corner detection effect using OpenCV.

Detects corners using the Shi-Tomasi (goodFeaturesToTrack) algorithm.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ShiTomasiCornersEffect(BaseUIEffect):
    """Detect and mark corners using Shi-Tomasi algorithm"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_corners = None

        # goodFeaturesToTrack parameters
        self.max_corners = tk.IntVar(value=100)
        self.quality_level = tk.DoubleVar(value=0.01)
        self.min_distance = tk.IntVar(value=10)
        self.block_size = tk.IntVar(value=3)

        # Drawing options
        self.corner_color_b = tk.IntVar(value=0)
        self.corner_color_g = tk.IntVar(value=255)
        self.corner_color_r = tk.IntVar(value=0)
        self.marker_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Detect Corners Shi-Tomasi"

    @classmethod
    def get_description(cls) -> str:
        return "Detect corners using Shi-Tomasi (goodFeaturesToTrack)"

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
            text="Shi-Tomasi Corner Detection",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)",
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

        # Max corners
        maxcorners_frame = ttk.Frame(right_column)
        maxcorners_frame.pack(fill='x', pady=3)

        ttk.Label(maxcorners_frame, text="Max Corners:").pack(side='left')

        maxcorners_slider = ttk.Scale(
            maxcorners_frame,
            from_=1,
            to=500,
            orient='horizontal',
            variable=self.max_corners,
            command=self._on_maxcorners_change
        )
        maxcorners_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.maxcorners_label = ttk.Label(maxcorners_frame, text="100")
        self.maxcorners_label.pack(side='left', padx=5)

        # Quality level
        quality_frame = ttk.Frame(right_column)
        quality_frame.pack(fill='x', pady=3)

        ttk.Label(quality_frame, text="Quality Level:").pack(side='left')

        quality_slider = ttk.Scale(
            quality_frame,
            from_=0.001,
            to=0.1,
            orient='horizontal',
            variable=self.quality_level,
            command=self._on_quality_change
        )
        quality_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.quality_label = ttk.Label(quality_frame, text="0.010")
        self.quality_label.pack(side='left', padx=5)

        # Min distance
        mindist_frame = ttk.Frame(right_column)
        mindist_frame.pack(fill='x', pady=3)

        ttk.Label(mindist_frame, text="Min Distance:").pack(side='left')

        mindist_slider = ttk.Scale(
            mindist_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.min_distance,
            command=self._on_mindist_change
        )
        mindist_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.mindist_label = ttk.Label(mindist_frame, text="10")
        self.mindist_label.pack(side='left', padx=5)

        # Block size
        block_frame = ttk.Frame(right_column)
        block_frame.pack(fill='x', pady=3)

        ttk.Label(block_frame, text="Block Size:").pack(side='left')

        block_slider = ttk.Scale(
            block_frame,
            from_=3,
            to=15,
            orient='horizontal',
            variable=self.block_size,
            command=self._on_block_change
        )
        block_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.block_label = ttk.Label(block_frame, text="3")
        self.block_label.pack(side='left', padx=5)

        # Corner color
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Corner Color (BGR):").pack(side='left')

        ttk.Label(color_frame, text="B:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_b).pack(side='left')

        ttk.Label(color_frame, text="G:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_g).pack(side='left')

        ttk.Label(color_frame, text="R:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_r).pack(side='left')

        # Marker size
        marker_frame = ttk.Frame(right_column)
        marker_frame.pack(fill='x', pady=3)

        ttk.Label(marker_frame, text="Marker Size:").pack(side='left')

        marker_slider = ttk.Scale(
            marker_frame,
            from_=1,
            to=15,
            orient='horizontal',
            variable=self.marker_size,
            command=self._on_marker_change
        )
        marker_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.marker_label = ttk.Label(marker_frame, text="5")
        self.marker_label.pack(side='left', padx=5)

        # Corners found display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Corners found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_maxcorners_change(self, value):
        self.maxcorners_label.config(text=str(int(float(value))))

    def _on_quality_change(self, value):
        self.quality_label.config(text=f"{float(value):.3f}")

    def _on_mindist_change(self, value):
        self.mindist_label.config(text=str(int(float(value))))

    def _on_block_change(self, value):
        self.block_label.config(text=str(int(float(value))))

    def _on_marker_change(self, value):
        self.marker_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and mark corners on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        max_corners = self.max_corners.get()
        quality_level = self.quality_level.get()
        min_distance = self.min_distance.get()
        block_size = self.block_size.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.corner_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.corner_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.corner_color_r.get()
        except Exception:
            color_r = 0
        color = (color_b, color_g, color_r)

        marker_size = self.marker_size.get()

        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            max_corners,
            quality_level,
            min_distance,
            blockSize=block_size
        )

        # Store raw values for pipeline use
        self.detected_corners = corners

        corners_count = 0

        if corners is not None:
            corners_count = len(corners)

            # Draw corners
            if self.draw_features.get():
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (int(x), int(y)), marker_size, color, -1)

        # Update count
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(corners_count))

        return result

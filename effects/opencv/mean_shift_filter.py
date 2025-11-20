"""
Pyramid Mean Shift Filtering effect using OpenCV.

Performs color segmentation producing a posterization-like effect.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class MeanShiftFilterEffect(BaseUIEffect):
    """Apply pyramid mean shift filtering for color segmentation"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Mean shift parameters
        self.spatial_radius = tk.IntVar(value=20)  # Spatial window radius
        self.color_radius = tk.IntVar(value=40)  # Color window radius
        self.max_level = tk.IntVar(value=1)  # Max pyramid level

    @classmethod
    def get_name(cls) -> str:
        return "Mean Shift Filter"

    @classmethod
    def get_description(cls) -> str:
        return "Color segmentation via pyramid mean shift"

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
            text="Mean Shift Filter",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.pyrMeanShiftFiltering(src, sp, sr, maxLevel)",
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

        # Spatial radius
        spatial_frame = ttk.Frame(right_column)
        spatial_frame.pack(fill='x', pady=3)

        ttk.Label(spatial_frame, text="Spatial Radius:").pack(side='left')

        spatial_slider = ttk.Scale(
            spatial_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.spatial_radius,
            command=self._on_spatial_change
        )
        spatial_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.spatial_label = ttk.Label(spatial_frame, text="20")
        self.spatial_label.pack(side='left', padx=5)

        # Color radius
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Color Radius:").pack(side='left')

        color_slider = ttk.Scale(
            color_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.color_radius,
            command=self._on_color_change
        )
        color_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.color_label = ttk.Label(color_frame, text="40")
        self.color_label.pack(side='left', padx=5)

        # Max pyramid level
        level_frame = ttk.Frame(right_column)
        level_frame.pack(fill='x', pady=3)

        ttk.Label(level_frame, text="Max Level:").pack(side='left')

        level_slider = ttk.Scale(
            level_frame,
            from_=0,
            to=4,
            orient='horizontal',
            variable=self.max_level,
            command=self._on_level_change
        )
        level_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.level_label = ttk.Label(level_frame, text="1")
        self.level_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_spatial_change(self, value):
        self.spatial_label.config(text=str(int(float(value))))

    def _on_color_change(self, value):
        self.color_label.config(text=str(int(float(value))))

    def _on_level_change(self, value):
        self.level_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply pyramid mean shift filtering to the frame"""
        if not self.enabled.get():
            return frame

        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Get parameters
        sp = self.spatial_radius.get()
        sr = self.color_radius.get()
        max_level = self.max_level.get()

        # Apply mean shift filtering
        result = cv2.pyrMeanShiftFiltering(frame, sp, sr, maxLevel=max_level)

        return result

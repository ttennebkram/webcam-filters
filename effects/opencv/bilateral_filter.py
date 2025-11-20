"""
Bilateral filter effect using OpenCV.

Edge-preserving smoothing that reduces noise while keeping edges sharp.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class BilateralFilterEffect(BaseUIEffect):
    """Apply bilateral filtering for edge-preserving smoothing"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Bilateral filter parameters
        self.diameter = tk.IntVar(value=9)  # Pixel neighborhood diameter
        self.sigma_color = tk.IntVar(value=75)  # Filter sigma in color space
        self.sigma_space = tk.IntVar(value=75)  # Filter sigma in coordinate space

    @classmethod
    def get_name(cls) -> str:
        return "Bilateral Filter"

    @classmethod
    def get_description(cls) -> str:
        return "Edge-preserving smoothing filter"

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
            text="Bilateral Filter",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)",
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

        # Diameter
        diameter_frame = ttk.Frame(right_column)
        diameter_frame.pack(fill='x', pady=3)

        ttk.Label(diameter_frame, text="Diameter:").pack(side='left')

        diameter_slider = ttk.Scale(
            diameter_frame,
            from_=1,
            to=25,
            orient='horizontal',
            variable=self.diameter,
            command=self._on_diameter_change
        )
        diameter_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.diameter_label = ttk.Label(diameter_frame, text="9")
        self.diameter_label.pack(side='left', padx=5)

        # Sigma Color
        sigma_color_frame = ttk.Frame(right_column)
        sigma_color_frame.pack(fill='x', pady=3)

        ttk.Label(sigma_color_frame, text="Sigma Color:").pack(side='left')

        sigma_color_slider = ttk.Scale(
            sigma_color_frame,
            from_=1,
            to=200,
            orient='horizontal',
            variable=self.sigma_color,
            command=self._on_sigma_color_change
        )
        sigma_color_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.sigma_color_label = ttk.Label(sigma_color_frame, text="75")
        self.sigma_color_label.pack(side='left', padx=5)

        # Sigma Space
        sigma_space_frame = ttk.Frame(right_column)
        sigma_space_frame.pack(fill='x', pady=3)

        ttk.Label(sigma_space_frame, text="Sigma Space:").pack(side='left')

        sigma_space_slider = ttk.Scale(
            sigma_space_frame,
            from_=1,
            to=200,
            orient='horizontal',
            variable=self.sigma_space,
            command=self._on_sigma_space_change
        )
        sigma_space_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.sigma_space_label = ttk.Label(sigma_space_frame, text="75")
        self.sigma_space_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_diameter_change(self, value):
        self.diameter_label.config(text=str(int(float(value))))

    def _on_sigma_color_change(self, value):
        self.sigma_color_label.config(text=str(int(float(value))))

    def _on_sigma_space_change(self, value):
        self.sigma_space_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply bilateral filter to the frame"""
        if not self.enabled.get():
            return frame

        # Get parameters
        diameter = self.diameter.get()
        sigma_color = self.sigma_color.get()
        sigma_space = self.sigma_space.get()

        # Apply bilateral filter
        result = cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)

        return result

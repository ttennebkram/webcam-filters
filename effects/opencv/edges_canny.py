"""
Canny edge detection effect using OpenCV.

Applies Canny edge detection algorithm to find edges in images.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class CannyEffect(BaseUIEffect):
    """Apply Canny edge detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.threshold1 = tk.IntVar(value=30)  # Lower threshold
        self.threshold2 = tk.IntVar(value=150)  # Upper threshold
        self.aperture_size = tk.IntVar(value=3)  # Sobel aperture size
        self.l2_gradient = tk.BooleanVar(value=False)  # Use L2 norm for gradient

    @classmethod
    def get_name(cls) -> str:
        return "Canny Edge Detection"

    @classmethod
    def get_description(cls) -> str:
        return "Apply Canny edge detection algorithm"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.Canny(image, threshold1, threshold2)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def create_control_panel(self, parent):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section (skip if in pipeline - LabelFrame already shows name)
        if not getattr(self, '_in_pipeline', False):
            header_frame = ttk.Frame(self.control_panel)
            header_frame.pack(fill='x', **padding)

            # Title in section header font
            title_label = ttk.Label(
                header_frame,
                text="Canny Edge Detection",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature for reference
            signature_label = ttk.Label(
                header_frame,
                text="cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)",
                font=('TkFixedFont', 12)
            )
            signature_label.pack(anchor='w', pady=(2, 2))

        # Main frame with two columns
        main_frame = ttk.Frame(self.control_panel)
        main_frame.pack(fill='x', **padding)

        # Left column - Enabled checkbox (vertically centered)
        left_column = ttk.Frame(main_frame)
        left_column.pack(side='left', fill='y', padx=(0, 15))

        # Spacer to center the checkbox vertically
        ttk.Frame(left_column).pack(expand=True)

        enabled_cb = ttk.Checkbutton(
            left_column,
            text="Enabled",
            variable=self.enabled
        )
        enabled_cb.pack()

        # Spacer below
        ttk.Frame(left_column).pack(expand=True)

        # Right column - all parameter controls
        right_column = ttk.Frame(main_frame)
        right_column.pack(side='left', fill='both', expand=True)

        # Threshold 1 (lower) control
        thresh1_frame = ttk.Frame(right_column)
        thresh1_frame.pack(fill='x', pady=3)

        ttk.Label(thresh1_frame, text="Threshold 1 (lower):").pack(side='left')

        thresh1_slider = ttk.Scale(
            thresh1_frame,
            from_=0,
            to=500,
            orient='horizontal',
            variable=self.threshold1,
            command=self._on_thresh1_change
        )
        thresh1_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh1_label = ttk.Label(thresh1_frame, text="30")
        self.thresh1_label.pack(side='left', padx=5)

        # Threshold 2 (upper) control
        thresh2_frame = ttk.Frame(right_column)
        thresh2_frame.pack(fill='x', pady=3)

        ttk.Label(thresh2_frame, text="Threshold 2 (upper):").pack(side='left')

        thresh2_slider = ttk.Scale(
            thresh2_frame,
            from_=0,
            to=500,
            orient='horizontal',
            variable=self.threshold2,
            command=self._on_thresh2_change
        )
        thresh2_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh2_label = ttk.Label(thresh2_frame, text="150")
        self.thresh2_label.pack(side='left', padx=5)

        # Aperture size dropdown
        aperture_frame = ttk.Frame(right_column)
        aperture_frame.pack(fill='x', pady=3)

        ttk.Label(aperture_frame, text="Aperture Size:").pack(side='left')

        aperture_values = [3, 5, 7]
        self.aperture_combo = ttk.Combobox(
            aperture_frame,
            values=aperture_values,
            state='readonly',
            width=5
        )
        self.aperture_combo.current(0)  # Default to 3
        self.aperture_combo.pack(side='left', padx=5)
        self.aperture_combo.bind('<<ComboboxSelected>>', self._on_aperture_change)

        ttk.Label(aperture_frame, text="(3, 5, or 7)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        # L2 gradient checkbox
        l2_frame = ttk.Frame(right_column)
        l2_frame.pack(fill='x', pady=3)

        l2_cb = ttk.Checkbutton(
            l2_frame,
            text="L2 Gradient",
            variable=self.l2_gradient
        )
        l2_cb.pack(side='left')

        ttk.Label(l2_frame, text="(more accurate, slower)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        return self.control_panel

    def _on_thresh1_change(self, value):
        """Handle threshold 1 slider change"""
        thresh = int(float(value))
        self.thresh1_label.config(text=str(thresh))

    def _on_thresh2_change(self, value):
        """Handle threshold 2 slider change"""
        thresh = int(float(value))
        self.thresh2_label.config(text=str(thresh))

    def _on_aperture_change(self, event):
        """Handle aperture size change"""
        self.aperture_size.set(int(self.aperture_combo.get()))

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Threshold 1: {self.threshold1.get()}")
        lines.append(f"Threshold 2: {self.threshold2.get()}")
        lines.append(f"Aperture Size: {self.aperture_size.get()}")
        lines.append(f"L2 Gradient: {'Yes' if self.l2_gradient.get() else 'No'}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Canny edge detection to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Get parameters
        threshold1 = self.threshold1.get()
        threshold2 = self.threshold2.get()
        aperture_size = self.aperture_size.get()
        l2_gradient = self.l2_gradient.get()

        # Apply Canny edge detection
        edges = cv2.Canny(
            gray,
            threshold1,
            threshold2,
            apertureSize=aperture_size,
            L2gradient=l2_gradient
        )

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return result

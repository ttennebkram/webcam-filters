"""
Scharr edge detection effect using OpenCV.

Scharr is similar to Sobel but optimized for 3x3 kernels with better rotational symmetry.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ScharrEffect(BaseUIEffect):
    """Apply Scharr edge detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Direction
        self.direction = tk.StringVar(value="both")  # x, y, or both

        # Scale and delta
        self.scale = tk.DoubleVar(value=1.0)
        self.delta = tk.IntVar(value=0)

        # Output depth
        self.use_absolute = tk.BooleanVar(value=True)

    @classmethod
    def get_name(cls) -> str:
        return "Edges Scharr"

    @classmethod
    def get_description(cls) -> str:
        return "Scharr edge detection (optimized 3x3 gradient)"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.Scharr(src, ddepth, dx, dy)"

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

            # Title
            title_label = ttk.Label(
                header_frame,
                text="Scharr Edge Detection",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.Scharr(src, ddepth, dx, dy, scale, delta)",
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

        # Direction
        dir_frame = ttk.Frame(right_column)
        dir_frame.pack(fill='x', pady=3)

        ttk.Label(dir_frame, text="Direction:").pack(side='left')

        ttk.Radiobutton(
            dir_frame,
            text="X",
            variable=self.direction,
            value="x"
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            dir_frame,
            text="Y",
            variable=self.direction,
            value="y"
        ).pack(side='left', padx=5)

        ttk.Radiobutton(
            dir_frame,
            text="Both",
            variable=self.direction,
            value="both"
        ).pack(side='left', padx=5)

        # Scale
        scale_frame = ttk.Frame(right_column)
        scale_frame.pack(fill='x', pady=3)

        ttk.Label(scale_frame, text="Scale:").pack(side='left')

        scale_slider = ttk.Scale(
            scale_frame,
            from_=0.1,
            to=5.0,
            orient='horizontal',
            variable=self.scale,
            command=self._on_scale_change
        )
        scale_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.scale_label = ttk.Label(scale_frame, text="1.0")
        self.scale_label.pack(side='left', padx=5)

        # Delta
        delta_frame = ttk.Frame(right_column)
        delta_frame.pack(fill='x', pady=3)

        ttk.Label(delta_frame, text="Delta:").pack(side='left')

        delta_slider = ttk.Scale(
            delta_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.delta,
            command=self._on_delta_change
        )
        delta_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.delta_label = ttk.Label(delta_frame, text="0")
        self.delta_label.pack(side='left', padx=5)

        # Absolute value checkbox
        abs_frame = ttk.Frame(right_column)
        abs_frame.pack(fill='x', pady=3)

        ttk.Checkbutton(
            abs_frame,
            text="Use Absolute Value",
            variable=self.use_absolute
        ).pack(side='left')

        return self.control_panel

    def _on_scale_change(self, value):
        self.scale_label.config(text=f"{float(value):.1f}")

    def _on_delta_change(self, value):
        self.delta_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Scharr edge detection to the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Get parameters
        scale = self.scale.get()
        delta = self.delta.get()
        direction = self.direction.get()

        if direction == "x":
            result = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
        elif direction == "y":
            result = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
        else:  # both
            scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale, delta=delta)
            scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale, delta=delta)
            result = cv2.magnitude(scharr_x, scharr_y)

        # Convert to absolute and 8-bit
        if self.use_absolute.get():
            result = np.absolute(result)

        result = np.uint8(np.clip(result, 0, 255))

        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

"""
Sobel gradient effect using OpenCV.

Computes image gradients using Sobel operator to detect edges.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class GradientSobelEffect(BaseUIEffect):
    """Compute image gradients using Sobel operator"""

    # Depth options for output
    DEPTH_OPTIONS = [
        (cv2.CV_8U, "CV_8U"),
        (cv2.CV_16S, "CV_16S"),
        (cv2.CV_32F, "CV_32F"),
        (cv2.CV_64F, "CV_64F"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.dx = tk.IntVar(value=1)  # Order of derivative in x
        self.dy = tk.IntVar(value=0)  # Order of derivative in y
        self.ksize = tk.IntVar(value=3)  # Kernel size (1, 3, 5, 7)
        self.depth_index = tk.IntVar(value=3)  # Default to CV_64F
        self.scale = tk.DoubleVar(value=1.0)
        self.delta = tk.DoubleVar(value=0.0)
        self.combine_xy = tk.BooleanVar(value=False)  # Combine gX and gY

    @classmethod
    def get_name(cls) -> str:
        return "Gradient (Sobel)"

    @classmethod
    def get_description(cls) -> str:
        return "Compute image gradients using Sobel operator for edge detection"

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

        # Title in section header font
        title_label = ttk.Label(
            header_frame,
            text="Sobel Gradient",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature for reference
        signature_label = ttk.Label(
            header_frame,
            text="cv2.Sobel(src, ddepth, dx, dy, ksize, scale, delta)",
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

        # Depth dropdown
        depth_frame = ttk.Frame(right_column)
        depth_frame.pack(fill='x', pady=3)

        ttk.Label(depth_frame, text="Depth (ddepth):").pack(side='left')

        depth_names = [name for _, name in self.DEPTH_OPTIONS]
        self.depth_combo = ttk.Combobox(
            depth_frame,
            values=depth_names,
            state='readonly',
            width=10
        )
        self.depth_combo.current(3)  # CV_64F
        self.depth_combo.pack(side='left', padx=5)
        self.depth_combo.bind('<<ComboboxSelected>>', self._on_depth_change)

        # dx control - radio buttons
        dx_frame = ttk.Frame(right_column)
        dx_frame.pack(fill='x', pady=3)

        ttk.Label(dx_frame, text="dx (x derivative):").pack(side='left')

        for val in [0, 1, 2]:
            ttk.Radiobutton(dx_frame, text=str(val), variable=self.dx, value=val).pack(side='left', padx=5)

        # dy control - radio buttons
        dy_frame = ttk.Frame(right_column)
        dy_frame.pack(fill='x', pady=3)

        ttk.Label(dy_frame, text="dy (y derivative):").pack(side='left')

        for val in [0, 1, 2]:
            ttk.Radiobutton(dy_frame, text=str(val), variable=self.dy, value=val).pack(side='left', padx=5)

        # ksize control
        ksize_frame = ttk.Frame(right_column)
        ksize_frame.pack(fill='x', pady=3)

        ttk.Label(ksize_frame, text="Kernel Size:").pack(side='left')

        # ksize must be 1, 3, 5, or 7
        ksize_values = [1, 3, 5, 7]
        self.ksize_combo = ttk.Combobox(
            ksize_frame,
            values=ksize_values,
            state='readonly',
            width=5
        )
        self.ksize_combo.current(1)  # Default to 3
        self.ksize_combo.pack(side='left', padx=5)
        self.ksize_combo.bind('<<ComboboxSelected>>', self._on_ksize_change)

        ttk.Label(ksize_frame, text="(1, 3, 5, or 7)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        # Scale control
        scale_frame = ttk.Frame(right_column)
        scale_frame.pack(fill='x', pady=3)

        ttk.Label(scale_frame, text="Scale:").pack(side='left')

        scale_slider = ttk.Scale(
            scale_frame,
            from_=0.1,
            to=10,
            orient='horizontal',
            variable=self.scale,
            command=self._on_scale_change
        )
        scale_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.scale_label = ttk.Label(scale_frame, text="1.0")
        self.scale_label.pack(side='left', padx=5)

        # Delta control
        delta_frame = ttk.Frame(right_column)
        delta_frame.pack(fill='x', pady=3)

        ttk.Label(delta_frame, text="Delta:").pack(side='left')

        delta_slider = ttk.Scale(
            delta_frame,
            from_=-128,
            to=128,
            orient='horizontal',
            variable=self.delta,
            command=self._on_delta_change
        )
        delta_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.delta_label = ttk.Label(delta_frame, text="0.0")
        self.delta_label.pack(side='left', padx=5)

        # Combine X and Y checkbox
        combine_frame = ttk.Frame(right_column)
        combine_frame.pack(fill='x', pady=3)

        combine_cb = ttk.Checkbutton(
            combine_frame,
            text="Combine gX and gY (magnitude)",
            variable=self.combine_xy
        )
        combine_cb.pack(side='left')

        return self.control_panel

    def _on_depth_change(self, event):
        """Handle depth change"""
        self.depth_index.set(self.depth_combo.current())

    def _on_ksize_change(self, event):
        """Handle ksize change"""
        self.ksize.set(int(self.ksize_combo.get()))

    def _on_scale_change(self, value):
        """Handle scale slider change"""
        scale = float(value)
        self.scale_label.config(text=f"{scale:.1f}")

    def _on_delta_change(self, value):
        """Handle delta slider change"""
        delta = float(value)
        self.delta_label.config(text=f"{delta:.1f}")

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Sobel gradient to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Get parameters
        ddepth, _ = self.DEPTH_OPTIONS[self.depth_index.get()]
        dx = int(self.dx.get())
        dy = int(self.dy.get())
        ksize = self.ksize.get()
        scale = self.scale.get()
        delta = self.delta.get()

        # Ensure at least one derivative is non-zero
        if dx == 0 and dy == 0:
            dx = 1

        if self.combine_xy.get():
            # Compute both gradients and combine
            gX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute magnitude
            magnitude = np.sqrt(gX**2 + gY**2)

            # Normalize to 0-255
            result = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)
        else:
            # Single Sobel operation
            gradient = cv2.Sobel(gray, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize, scale=scale, delta=delta)

            # Convert to displayable format
            if ddepth in [cv2.CV_64F, cv2.CV_32F, cv2.CV_16S]:
                # Take absolute value and normalize
                result = cv2.convertScaleAbs(gradient)
            else:
                result = gradient

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

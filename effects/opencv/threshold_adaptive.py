"""
Adaptive threshold effect using OpenCV.

Applies adaptive thresholding which calculates threshold for small regions.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ThresholdAdaptiveEffect(BaseUIEffect):
    """Apply adaptive thresholding to create binary images"""

    # Adaptive method options
    ADAPTIVE_METHODS = [
        (cv2.ADAPTIVE_THRESH_MEAN_C, "ADAPTIVE_THRESH_MEAN_C"),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "ADAPTIVE_THRESH_GAUSSIAN_C"),
    ]

    # Threshold type options
    THRESHOLD_TYPES = [
        (cv2.THRESH_BINARY, "THRESH_BINARY"),
        (cv2.THRESH_BINARY_INV, "THRESH_BINARY_INV"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.max_value = tk.IntVar(value=255)
        self.adaptive_method_index = tk.IntVar(value=0)  # ADAPTIVE_THRESH_MEAN_C
        self.thresh_type_index = tk.IntVar(value=0)  # THRESH_BINARY
        self.block_size = tk.IntVar(value=25)  # Must be odd
        self.c_value = tk.IntVar(value=15)  # Constant subtracted from mean

    @classmethod
    def get_name(cls) -> str:
        return "Threshold (Adaptive)"

    @classmethod
    def get_description(cls) -> str:
        return "Apply adaptive thresholding with local region calculation"

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
                text="Adaptive Threshold",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature for reference
            signature_label = ttk.Label(
                header_frame,
                text="cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)",
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

        # Max value control
        max_frame = ttk.Frame(right_column)
        max_frame.pack(fill='x', pady=3)

        ttk.Label(max_frame, text="Max Value:").pack(side='left')

        max_slider = ttk.Scale(
            max_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.max_value,
            command=self._on_max_change
        )
        max_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.max_label = ttk.Label(max_frame, text="255")
        self.max_label.pack(side='left', padx=5)

        # Adaptive method dropdown
        method_frame = ttk.Frame(right_column)
        method_frame.pack(fill='x', pady=3)

        ttk.Label(method_frame, text="Adaptive Method:").pack(side='left')

        method_names = [name for _, name in self.ADAPTIVE_METHODS]
        self.method_combo = ttk.Combobox(
            method_frame,
            values=method_names,
            state='readonly',
            width=25
        )
        self.method_combo.current(0)
        self.method_combo.pack(side='left', padx=5)
        self.method_combo.bind('<<ComboboxSelected>>', self._on_method_change)

        # Threshold type dropdown
        type_frame = ttk.Frame(right_column)
        type_frame.pack(fill='x', pady=3)

        ttk.Label(type_frame, text="Threshold Type:").pack(side='left')

        type_names = [name for _, name in self.THRESHOLD_TYPES]
        self.type_combo = ttk.Combobox(
            type_frame,
            values=type_names,
            state='readonly',
            width=20
        )
        self.type_combo.current(0)
        self.type_combo.pack(side='left', padx=5)
        self.type_combo.bind('<<ComboboxSelected>>', self._on_type_change)

        # Block size control
        block_frame = ttk.Frame(right_column)
        block_frame.pack(fill='x', pady=3)

        ttk.Label(block_frame, text="Block Size:").pack(side='left')

        block_slider = ttk.Scale(
            block_frame,
            from_=3,
            to=99,
            orient='horizontal',
            variable=self.block_size,
            command=self._on_block_change
        )
        block_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.block_label = ttk.Label(block_frame, text="25")
        self.block_label.pack(side='left', padx=5)

        ttk.Label(block_frame, text="(must be odd)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        # C value control
        c_frame = ttk.Frame(right_column)
        c_frame.pack(fill='x', pady=3)

        ttk.Label(c_frame, text="C (constant):").pack(side='left')

        c_slider = ttk.Scale(
            c_frame,
            from_=-50,
            to=50,
            orient='horizontal',
            variable=self.c_value,
            command=self._on_c_change
        )
        c_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.c_label = ttk.Label(c_frame, text="15")
        self.c_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_max_change(self, value):
        """Handle max value slider change"""
        maxval = int(float(value))
        self.max_label.config(text=str(maxval))

    def _on_method_change(self, event):
        """Handle adaptive method change"""
        self.adaptive_method_index.set(self.method_combo.current())

    def _on_type_change(self, event):
        """Handle threshold type change"""
        self.thresh_type_index.set(self.type_combo.current())

    def _on_block_change(self, value):
        """Handle block size slider change - ensure odd value"""
        block = int(float(value))
        # Ensure block size is odd
        if block % 2 == 0:
            block = block + 1
        self.block_size.set(block)
        self.block_label.config(text=str(block))

    def _on_c_change(self, value):
        """Handle C value slider change"""
        c = int(float(value))
        self.c_label.config(text=str(c))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply adaptive thresholding to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get parameters
        maxval = self.max_value.get()
        adaptive_method, _ = self.ADAPTIVE_METHODS[self.adaptive_method_index.get()]
        thresh_type, _ = self.THRESHOLD_TYPES[self.thresh_type_index.get()]
        block_size = self.block_size.get()
        c_value = self.c_value.get()

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size = block_size + 1

        # Adaptive threshold requires grayscale input
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply adaptive threshold
        thresholded = cv2.adaptiveThreshold(
            gray, maxval, adaptive_method, thresh_type, block_size, c_value
        )

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        return result

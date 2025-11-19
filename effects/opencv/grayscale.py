"""
Color conversion effect using OpenCV.

Converts images between different color spaces using cv2.cvtColor.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class GrayscaleEffect(BaseUIEffect):
    """Convert image to different color spaces"""

    # Color conversion options (code, display name)
    COLOR_CONVERSIONS = [
        (cv2.COLOR_BGR2GRAY, "BGR to Grayscale"),
        (cv2.COLOR_BGR2RGB, "BGR to RGB"),
        (cv2.COLOR_BGR2HSV, "BGR to HSV"),
        (cv2.COLOR_BGR2HLS, "BGR to HLS"),
        (cv2.COLOR_BGR2LAB, "BGR to LAB"),
        (cv2.COLOR_BGR2LUV, "BGR to LUV"),
        (cv2.COLOR_BGR2YCrCb, "BGR to YCrCb"),
        (cv2.COLOR_BGR2XYZ, "BGR to XYZ"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.conversion_index = tk.IntVar(value=0)  # Index into COLOR_CONVERSIONS

    @classmethod
    def get_name(cls) -> str:
        return "Grayscale / Color Convert"

    @classmethod
    def get_description(cls) -> str:
        return "Convert image between different color spaces"

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
            text="Color Conversion",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature for reference
        signature_label = ttk.Label(
            header_frame,
            text="cv2.cvtColor(src, code)",
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

        # Right column - conversion dropdown
        right_column = ttk.Frame(main_frame)
        right_column.pack(side='left', fill='both', expand=True)

        # Conversion type dropdown
        conv_frame = ttk.Frame(right_column)
        conv_frame.pack(fill='x', pady=3)

        ttk.Label(conv_frame, text="Conversion:").pack(side='left')

        # Create dropdown with conversion names
        conversion_names = [name for _, name in self.COLOR_CONVERSIONS]
        self.conv_combo = ttk.Combobox(
            conv_frame,
            values=conversion_names,
            state='readonly',
            width=20
        )
        self.conv_combo.current(0)
        self.conv_combo.pack(side='left', padx=5)
        self.conv_combo.bind('<<ComboboxSelected>>', self._on_conversion_change)

        return self.control_panel

    def _on_conversion_change(self, event):
        """Handle conversion type change"""
        self.conversion_index.set(self.conv_combo.current())

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply color conversion to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get selected conversion
        conv_code, _ = self.COLOR_CONVERSIONS[self.conversion_index.get()]

        # Apply conversion
        converted = cv2.cvtColor(frame, conv_code)

        # If result is grayscale, convert back to BGR for display
        if len(converted.shape) == 2:
            converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)

        return converted

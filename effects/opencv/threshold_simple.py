"""
Simple threshold effect using OpenCV.

Applies basic thresholding to convert images to binary.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ThresholdSimpleEffect(BaseUIEffect):
    """Apply simple thresholding to create binary images"""

    # Threshold type options (code, display name)
    THRESHOLD_TYPES = [
        (cv2.THRESH_BINARY, "THRESH_BINARY"),
        (cv2.THRESH_BINARY_INV, "THRESH_BINARY_INV"),
        (cv2.THRESH_TRUNC, "THRESH_TRUNC"),
        (cv2.THRESH_TOZERO, "THRESH_TOZERO"),
        (cv2.THRESH_TOZERO_INV, "THRESH_TOZERO_INV"),
    ]

    # Modifier flags (can be OR'd with basic types)
    THRESHOLD_MODIFIERS = [
        (0, "None"),
        (cv2.THRESH_OTSU, "THRESH_OTSU"),
        (cv2.THRESH_TRIANGLE, "THRESH_TRIANGLE"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.thresh_value = tk.IntVar(value=200)  # Threshold value
        self.max_value = tk.IntVar(value=255)  # Maximum value
        self.thresh_type_index = tk.IntVar(value=0)  # Default to THRESH_BINARY
        self.thresh_modifier_index = tk.IntVar(value=0)  # Default to None

    @classmethod
    def get_name(cls) -> str:
        return "Threshold (Simple)"

    @classmethod
    def get_description(cls) -> str:
        return "Apply simple thresholding to create binary images"

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
                text="Simple Threshold",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature for reference
            signature_label = ttk.Label(
                header_frame,
                text="cv2.threshold(src, thresh, maxval, type)",
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

        # Threshold value control
        thresh_frame = ttk.Frame(right_column)
        thresh_frame.pack(fill='x', pady=3)

        ttk.Label(thresh_frame, text="Threshold:").pack(side='left')

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.thresh_value,
            command=self._on_thresh_change
        )
        thresh_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh_label = ttk.Label(thresh_frame, text="200")
        self.thresh_label.pack(side='left', padx=5)

        # Note about modifier ignoring threshold
        ttk.Label(thresh_frame, text="(ignored w/ modifier)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

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

        # Threshold type dropdown
        type_frame = ttk.Frame(right_column)
        type_frame.pack(fill='x', pady=3)

        ttk.Label(type_frame, text="Type:").pack(side='left')

        # Create dropdown with type names
        type_names = [name for _, name in self.THRESHOLD_TYPES]
        self.type_combo = ttk.Combobox(
            type_frame,
            values=type_names,
            state='readonly',
            width=20
        )
        self.type_combo.current(0)  # Default to THRESH_BINARY
        self.type_combo.pack(side='left', padx=5)
        self.type_combo.bind('<<ComboboxSelected>>', self._on_type_change)

        # Bitwise OR label (italic)
        ttk.Label(type_frame, text="(bitwise OR)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        # Modifier dropdown (OTSU, TRIANGLE)
        modifier_frame = ttk.Frame(right_column)
        modifier_frame.pack(fill='x', pady=3)

        ttk.Label(modifier_frame, text="Modifier:").pack(side='left')

        # Create dropdown with modifier names
        modifier_names = [name for _, name in self.THRESHOLD_MODIFIERS]
        self.modifier_combo = ttk.Combobox(
            modifier_frame,
            values=modifier_names,
            state='readonly',
            width=20
        )
        self.modifier_combo.current(0)  # Default to None
        self.modifier_combo.pack(side='left', padx=5)
        self.modifier_combo.bind('<<ComboboxSelected>>', self._on_modifier_change)

        # Forces grayscale label (italic)
        ttk.Label(modifier_frame, text="(forces grayscale)", font=('TkDefaultFont', 10, 'italic')).pack(side='left', padx=5)

        # Calculated T value display
        t_frame = ttk.Frame(right_column)
        t_frame.pack(fill='x', pady=3)

        ttk.Label(t_frame, text="Calculated T:").pack(side='left')
        self.t_value_label = ttk.Label(t_frame, text="—")
        self.t_value_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_thresh_change(self, value):
        """Handle threshold value slider change"""
        thresh = int(float(value))
        self.thresh_label.config(text=str(thresh))

    def _on_max_change(self, value):
        """Handle max value slider change"""
        maxval = int(float(value))
        self.max_label.config(text=str(maxval))

    def _on_type_change(self, event):
        """Handle threshold type change"""
        self.thresh_type_index.set(self.type_combo.current())

    def _on_modifier_change(self, event):
        """Handle threshold modifier change"""
        self.thresh_modifier_index.set(self.modifier_combo.current())

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply thresholding to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get parameters
        thresh = self.thresh_value.get()
        maxval = self.max_value.get()
        thresh_type, _ = self.THRESHOLD_TYPES[self.thresh_type_index.get()]
        modifier, _ = self.THRESHOLD_MODIFIERS[self.thresh_modifier_index.get()]

        # Combine type with modifier using bitwise OR
        combined_type = thresh_type | modifier

        # OTSU and TRIANGLE require grayscale input
        if modifier in (cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE):
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply threshold
            t_value, thresholded = cv2.threshold(gray, thresh, maxval, combined_type)

            # Update T value display
            if hasattr(self, 't_value_label'):
                self.t_value_label.config(text=f"{t_value:.1f}")

            # Convert back to BGR (3 identical channels)
            result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            return result
        else:
            # Apply threshold (works on grayscale or each BGR channel separately)
            t_value, thresholded = cv2.threshold(frame, thresh, maxval, combined_type)

            # Update T value display (shows input thresh when no modifier)
            if hasattr(self, 't_value_label'):
                self.t_value_label.config(text=f"{t_value:.1f}")

            return thresholded

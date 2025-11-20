"""
Color inRange filtering effect using OpenCV.

Filters pixels by color range in HSV or BGR color space.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class InRangeEffect(BaseUIEffect):
    """Filter colors within a specified range"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Color space
        self.use_hsv = tk.BooleanVar(value=True)

        # HSV ranges (Hue: 0-179, Sat: 0-255, Val: 0-255)
        self.h_low = tk.IntVar(value=0)
        self.h_high = tk.IntVar(value=179)
        self.s_low = tk.IntVar(value=0)
        self.s_high = tk.IntVar(value=255)
        self.v_low = tk.IntVar(value=0)
        self.v_high = tk.IntVar(value=255)

        # Output mode
        self.output_mode = tk.StringVar(value="mask")  # mask, masked, inverse

    @classmethod
    def get_name(cls) -> str:
        return "Color InRange"

    @classmethod
    def get_description(cls) -> str:
        return "Filter colors within a specified range"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.inRange(src, lowerb, upperb)"

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
                text="Color InRange",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.inRange(src, lowerb, upperb)",
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

        # Color space selection
        space_frame = ttk.Frame(right_column)
        space_frame.pack(fill='x', pady=3)

        ttk.Label(space_frame, text="Color Space:").pack(side='left')

        ttk.Radiobutton(
            space_frame,
            text="HSV",
            variable=self.use_hsv,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            space_frame,
            text="BGR",
            variable=self.use_hsv,
            value=False
        ).pack(side='left', padx=5)

        # Hue/Blue low
        h_low_frame = ttk.Frame(right_column)
        h_low_frame.pack(fill='x', pady=3)

        ttk.Label(h_low_frame, text="H/B Low:").pack(side='left')

        h_low_slider = ttk.Scale(
            h_low_frame,
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.h_low,
            command=self._on_h_low_change
        )
        h_low_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.h_low_label = ttk.Label(h_low_frame, text="0")
        self.h_low_label.pack(side='left', padx=5)

        # Hue/Blue high
        h_high_frame = ttk.Frame(right_column)
        h_high_frame.pack(fill='x', pady=3)

        ttk.Label(h_high_frame, text="H/B High:").pack(side='left')

        h_high_slider = ttk.Scale(
            h_high_frame,
            from_=0,
            to=179,
            orient='horizontal',
            variable=self.h_high,
            command=self._on_h_high_change
        )
        h_high_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.h_high_label = ttk.Label(h_high_frame, text="179")
        self.h_high_label.pack(side='left', padx=5)

        # Saturation/Green low
        s_low_frame = ttk.Frame(right_column)
        s_low_frame.pack(fill='x', pady=3)

        ttk.Label(s_low_frame, text="S/G Low:").pack(side='left')

        s_low_slider = ttk.Scale(
            s_low_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.s_low,
            command=self._on_s_low_change
        )
        s_low_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.s_low_label = ttk.Label(s_low_frame, text="0")
        self.s_low_label.pack(side='left', padx=5)

        # Saturation/Green high
        s_high_frame = ttk.Frame(right_column)
        s_high_frame.pack(fill='x', pady=3)

        ttk.Label(s_high_frame, text="S/G High:").pack(side='left')

        s_high_slider = ttk.Scale(
            s_high_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.s_high,
            command=self._on_s_high_change
        )
        s_high_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.s_high_label = ttk.Label(s_high_frame, text="255")
        self.s_high_label.pack(side='left', padx=5)

        # Value/Red low
        v_low_frame = ttk.Frame(right_column)
        v_low_frame.pack(fill='x', pady=3)

        ttk.Label(v_low_frame, text="V/R Low:").pack(side='left')

        v_low_slider = ttk.Scale(
            v_low_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.v_low,
            command=self._on_v_low_change
        )
        v_low_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.v_low_label = ttk.Label(v_low_frame, text="0")
        self.v_low_label.pack(side='left', padx=5)

        # Value/Red high
        v_high_frame = ttk.Frame(right_column)
        v_high_frame.pack(fill='x', pady=3)

        ttk.Label(v_high_frame, text="V/R High:").pack(side='left')

        v_high_slider = ttk.Scale(
            v_high_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.v_high,
            command=self._on_v_high_change
        )
        v_high_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.v_high_label = ttk.Label(v_high_frame, text="255")
        self.v_high_label.pack(side='left', padx=5)

        # Output mode
        output_frame = ttk.Frame(right_column)
        output_frame.pack(fill='x', pady=3)

        ttk.Label(output_frame, text="Output:").pack(side='left')

        ttk.Radiobutton(
            output_frame,
            text="Mask",
            variable=self.output_mode,
            value="mask"
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            output_frame,
            text="Masked",
            variable=self.output_mode,
            value="masked"
        ).pack(side='left', padx=5)

        ttk.Radiobutton(
            output_frame,
            text="Inverse",
            variable=self.output_mode,
            value="inverse"
        ).pack(side='left', padx=5)

        return self.control_panel

    def _on_h_low_change(self, value):
        self.h_low_label.config(text=str(int(float(value))))

    def _on_h_high_change(self, value):
        self.h_high_label.config(text=str(int(float(value))))

    def _on_s_low_change(self, value):
        self.s_low_label.config(text=str(int(float(value))))

    def _on_s_high_change(self, value):
        self.s_high_label.config(text=str(int(float(value))))

    def _on_v_low_change(self, value):
        self.v_low_label.config(text=str(int(float(value))))

    def _on_v_high_change(self, value):
        self.v_high_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply inRange color filtering to the frame"""
        if not self.enabled.get():
            return frame

        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Convert to HSV if needed
        if self.use_hsv.get():
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            converted = frame

        # Get range values
        lower = np.array([self.h_low.get(), self.s_low.get(), self.v_low.get()])
        upper = np.array([self.h_high.get(), self.s_high.get(), self.v_high.get()])

        # Create mask
        mask = cv2.inRange(converted, lower, upper)

        # Apply output mode
        output_mode = self.output_mode.get()

        if output_mode == "mask":
            # Return binary mask as BGR
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif output_mode == "masked":
            # Apply mask to original image
            result = cv2.bitwise_and(frame, frame, mask=mask)
        else:  # inverse
            # Invert mask and apply
            inv_mask = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(frame, frame, mask=inv_mask)

        return result

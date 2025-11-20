"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) effect using OpenCV.

Enhances local contrast using adaptive histogram equalization with clipping.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class CLAHEEffect(BaseUIEffect):
    """Apply CLAHE for adaptive contrast enhancement"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.clip_limit = tk.DoubleVar(value=2.0)
        self.tile_size = tk.IntVar(value=8)
        self.color_mode = tk.StringVar(value="lab")  # lab, hsv, or grayscale

    @classmethod
    def get_name(cls) -> str:
        return "CLAHE: Contrast Enhancement"

    @classmethod
    def get_description(cls) -> str:
        return "CLAHE: Contrast Limited Adaptive Histogram Equalization"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.createCLAHE(clipLimit, tileGridSize)"

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
                text="CLAHE: Contrast Limited Adaptive Histogram Equalization",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            desc_label = ttk.Label(
                header_frame,
                text="cv2.createCLAHE(clipLimit, tileGridSize)",
                font=('TkFixedFont', 12)
            )
            desc_label.pack(anchor='w', pady=(2, 2))

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

        # Clip Limit slider
        clip_frame = ttk.Frame(right_column)
        clip_frame.pack(fill='x', pady=3)

        ttk.Label(clip_frame, text="Clip Limit:").pack(side='left')

        def on_clip_change(*args):
            self.clip_label.config(text=f"{self.clip_limit.get():.1f}")

        clip_slider = ttk.Scale(
            clip_frame,
            from_=1.0,
            to=40.0,
            orient='horizontal',
            variable=self.clip_limit,
            command=lambda v: on_clip_change()
        )
        clip_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.clip_label = ttk.Label(clip_frame, text="2.0", width=5)
        self.clip_label.pack(side='left', padx=5)

        self.clip_limit.trace_add("write", on_clip_change)

        # Tile Size slider
        tile_frame = ttk.Frame(right_column)
        tile_frame.pack(fill='x', pady=3)

        ttk.Label(tile_frame, text="Tile Size:").pack(side='left')

        def on_tile_change(*args):
            self.tile_label.config(text=f"{self.tile_size.get()}x{self.tile_size.get()}")

        tile_slider = ttk.Scale(
            tile_frame,
            from_=2,
            to=32,
            orient='horizontal',
            variable=self.tile_size,
            command=lambda v: on_tile_change()
        )
        tile_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.tile_label = ttk.Label(tile_frame, text="8x8", width=6)
        self.tile_label.pack(side='left', padx=5)

        self.tile_size.trace_add("write", on_tile_change)

        # Color mode selection
        mode_frame = ttk.Frame(right_column)
        mode_frame.pack(fill='x', pady=3)

        ttk.Label(mode_frame, text="Apply to:").pack(side='left')

        ttk.Radiobutton(
            mode_frame,
            text="LAB (L channel)",
            variable=self.color_mode,
            value="lab"
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            mode_frame,
            text="HSV (V channel)",
            variable=self.color_mode,
            value="hsv"
        ).pack(side='left', padx=5)

        ttk.Radiobutton(
            mode_frame,
            text="Grayscale",
            variable=self.color_mode,
            value="grayscale"
        ).pack(side='left', padx=5)

        return self.control_panel

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply CLAHE to the frame"""
        if not self.enabled.get():
            return frame

        clip_limit = self.clip_limit.get()
        tile_size = int(self.tile_size.get())
        color_mode = self.color_mode.get()

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )

        if color_mode == "grayscale":
            # Convert to grayscale, apply CLAHE, convert back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = clahe.apply(gray)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elif color_mode == "lab":
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif color_mode == "hsv":
            # Convert to HSV, apply CLAHE to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        else:
            result = frame

        return result

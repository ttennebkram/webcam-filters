"""
Median blur effect using OpenCV.

Noise reduction filter that replaces each pixel with the median of neighboring pixels.
Particularly effective at removing salt-and-pepper noise while preserving edges.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class MedianBlurEffect(BaseUIEffect):
    """Apply median blur for noise reduction"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Median blur parameter (must be odd)
        self.kernel_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Median Blur"

    @classmethod
    def get_description(cls) -> str:
        return "Noise reduction using median filter"

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
                text="Median Blur",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.medianBlur(src, ksize)",
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

        # Kernel size
        ksize_frame = ttk.Frame(right_column)
        ksize_frame.pack(fill='x', pady=3)

        ttk.Label(ksize_frame, text="Kernel Size:").pack(side='left')

        ksize_slider = ttk.Scale(
            ksize_frame,
            from_=1,
            to=31,
            orient='horizontal',
            variable=self.kernel_size,
            command=self._on_ksize_change
        )
        ksize_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ksize_label = ttk.Label(ksize_frame, text="5")
        self.ksize_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_ksize_change(self, value):
        """Handle kernel size slider change - ensure odd value"""
        ksize = int(float(value))
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size.set(ksize)
        self.ksize_label.config(text=str(ksize))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply median blur to the frame"""
        if not self.enabled.get():
            return frame

        # Get kernel size (must be odd)
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1

        # Apply median blur
        result = cv2.medianBlur(frame, ksize)

        return result

"""
Morphological dilation effect using OpenCV.

Dilates (expands) bright regions in the image.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class DilateEffect(BaseUIEffect):
    """Apply morphological dilation"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Kernel size (must be odd)
        self.kernel_size = tk.IntVar(value=5)

        # Kernel shape
        self.kernel_shape = tk.IntVar(value=0)  # Index into KERNEL_SHAPES

        # Iterations
        self.iterations = tk.IntVar(value=1)

    KERNEL_SHAPES = [
        (cv2.MORPH_RECT, "Rectangle"),
        (cv2.MORPH_ELLIPSE, "Ellipse"),
        (cv2.MORPH_CROSS, "Cross"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "Morph Dilate"

    @classmethod
    def get_description(cls) -> str:
        return "Morphological dilation - expands bright regions"

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
            text="Dilate",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.dilate(src, kernel, iterations)",
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

        # Kernel shape
        shape_frame = ttk.Frame(right_column)
        shape_frame.pack(fill='x', pady=3)

        ttk.Label(shape_frame, text="Kernel Shape:").pack(side='left')

        shape_values = [name for _, name in self.KERNEL_SHAPES]
        self.shape_combo = ttk.Combobox(
            shape_frame,
            values=shape_values,
            state='readonly',
            width=12
        )
        self.shape_combo.current(0)
        self.shape_combo.pack(side='left', padx=5)
        self.shape_combo.bind('<<ComboboxSelected>>', self._on_shape_change)

        # Iterations
        iter_frame = ttk.Frame(right_column)
        iter_frame.pack(fill='x', pady=3)

        ttk.Label(iter_frame, text="Iterations:").pack(side='left')

        iter_slider = ttk.Scale(
            iter_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.iterations,
            command=self._on_iter_change
        )
        iter_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.iter_label = ttk.Label(iter_frame, text="1")
        self.iter_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_ksize_change(self, value):
        """Handle kernel size slider change"""
        # Ensure odd value
        ksize = int(float(value))
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size.set(ksize)
        self.ksize_label.config(text=str(ksize))

    def _on_shape_change(self, event):
        """Handle kernel shape change"""
        self.kernel_shape.set(self.shape_combo.current())

    def _on_iter_change(self, value):
        """Handle iterations slider change"""
        self.iter_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply dilation to the frame"""
        if not self.enabled.get():
            return frame

        # Get parameters
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1

        shape_idx = self.kernel_shape.get()
        shape = self.KERNEL_SHAPES[shape_idx][0]

        iterations = self.iterations.get()

        # Create kernel
        kernel = cv2.getStructuringElement(shape, (ksize, ksize))

        # Apply dilation
        result = cv2.dilate(frame, kernel, iterations=iterations)

        return result

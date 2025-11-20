"""
Blur effect using OpenCV.

Applies Gaussian blur to the image with adjustable parameters.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class BlurEffect(BaseUIEffect):
    """Apply Gaussian blur effect to the image"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.kernel_size_x = tk.IntVar(value=7)  # Kernel width
        self.kernel_size_y = tk.IntVar(value=7)  # Kernel height
        self.sigma_x = tk.DoubleVar(value=0.0)  # 0 means calculated from kernel size

    @classmethod
    def get_name(cls) -> str:
        return "Blur"

    @classmethod
    def get_description(cls) -> str:
        return "Apply Gaussian blur effect to soften the image"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.GaussianBlur(src, ksize, sigmaX)"

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
                text="Gaussian Blur",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature for reference
            signature_label = ttk.Label(
                header_frame,
                text="cv2.GaussianBlur(src, ksize (x, y), sigmaX)",
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

        # Kernel Size X control
        ksize_x_frame = ttk.Frame(right_column)
        ksize_x_frame.pack(fill='x', pady=3)

        ttk.Label(ksize_x_frame, text="Kernel Size X:").pack(side='left')

        # Kernel size must be odd, so we use values 1, 3, 5, ..., 31
        ksize_x_slider = ttk.Scale(
            ksize_x_frame,
            from_=1,
            to=31,
            orient='horizontal',
            variable=self.kernel_size_x,
            command=self._on_kernel_x_change
        )
        ksize_x_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ksize_x_label = ttk.Label(ksize_x_frame, text="7")
        self.ksize_x_label.pack(side='left', padx=5)

        # Kernel Size Y control
        ksize_y_frame = ttk.Frame(right_column)
        ksize_y_frame.pack(fill='x', pady=3)

        ttk.Label(ksize_y_frame, text="Kernel Size Y:").pack(side='left')

        ksize_y_slider = ttk.Scale(
            ksize_y_frame,
            from_=1,
            to=31,
            orient='horizontal',
            variable=self.kernel_size_y,
            command=self._on_kernel_y_change
        )
        ksize_y_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ksize_y_label = ttk.Label(ksize_y_frame, text="7")
        self.ksize_y_label.pack(side='left', padx=5)

        # Sigma X control
        sigma_frame = ttk.Frame(right_column)
        sigma_frame.pack(fill='x', pady=3)

        ttk.Label(sigma_frame, text="Sigma X:").pack(side='left')

        sigma_slider = ttk.Scale(
            sigma_frame,
            from_=0,
            to=10,
            orient='horizontal',
            variable=self.sigma_x,
            command=self._on_sigma_change
        )
        sigma_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.sigma_label = ttk.Label(sigma_frame, text="0.0 (auto)")
        self.sigma_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_kernel_x_change(self, value):
        """Handle kernel size X slider change - ensure odd value"""
        ksize = int(float(value))
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize = ksize + 1
        self.kernel_size_x.set(ksize)
        self.ksize_x_label.config(text=str(ksize))

    def _on_kernel_y_change(self, value):
        """Handle kernel size Y slider change - ensure odd value"""
        ksize = int(float(value))
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize = ksize + 1
        self.kernel_size_y.set(ksize)
        self.ksize_y_label.config(text=str(ksize))

    def _on_sigma_change(self, value):
        """Handle sigma slider change"""
        sigma = float(value)
        if sigma == 0:
            self.sigma_label.config(text="0.0 (auto)")
        else:
            self.sigma_label.config(text=f"{sigma:.1f}")

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Gaussian blur to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get kernel sizes (ensure they're odd)
        ksize_x = self.kernel_size_x.get()
        if ksize_x % 2 == 0:
            ksize_x = ksize_x + 1

        ksize_y = self.kernel_size_y.get()
        if ksize_y % 2 == 0:
            ksize_y = ksize_y + 1

        # Get sigma
        sigma_x = self.sigma_x.get()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (ksize_x, ksize_y), sigma_x)

        return blurred

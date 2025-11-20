"""
Harris corner detection effect using OpenCV.

Detects corners using the Harris corner detection algorithm.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class HarrisCornersEffect(BaseUIEffect):
    """Detect and mark corners using Harris corner detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_corners = None

        # Harris parameters
        self.block_size = tk.IntVar(value=2)  # Neighborhood size
        self.ksize = tk.IntVar(value=3)  # Sobel aperture
        self.k = tk.DoubleVar(value=0.04)  # Harris free parameter

        # Threshold for corner detection
        self.threshold = tk.DoubleVar(value=0.01)  # Fraction of max response

        # Drawing options
        self.corner_color_b = tk.IntVar(value=0)
        self.corner_color_g = tk.IntVar(value=0)
        self.corner_color_r = tk.IntVar(value=255)
        self.marker_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Detect Corners Harris"

    @classmethod
    def get_description(cls) -> str:
        return "Detect corners using Harris corner detection"

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
            text="Harris Corner Detection",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.cornerHarris(src, blockSize, ksize, k)",
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

        # Input option
        input_frame = ttk.Frame(right_column)
        input_frame.pack(fill='x', pady=3)

        ttk.Label(input_frame, text="Input:").pack(side='left')

        ttk.Radiobutton(
            input_frame,
            text="Original Image",
            variable=self.show_original,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            input_frame,
            text="Black",
            variable=self.show_original,
            value=False
        ).pack(side='left', padx=5)

        # Output option
        output_frame = ttk.Frame(right_column)
        output_frame.pack(fill='x', pady=3)

        ttk.Label(output_frame, text="Output:").pack(side='left')

        ttk.Radiobutton(
            output_frame,
            text="Draw Features",
            variable=self.draw_features,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            output_frame,
            text="Raw Values Only",
            variable=self.draw_features,
            value=False
        ).pack(side='left', padx=5)

        # Block size
        block_frame = ttk.Frame(right_column)
        block_frame.pack(fill='x', pady=3)

        ttk.Label(block_frame, text="Block Size:").pack(side='left')

        block_slider = ttk.Scale(
            block_frame,
            from_=2,
            to=10,
            orient='horizontal',
            variable=self.block_size,
            command=self._on_block_change
        )
        block_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.block_label = ttk.Label(block_frame, text="2")
        self.block_label.pack(side='left', padx=5)

        # Sobel aperture size
        ksize_frame = ttk.Frame(right_column)
        ksize_frame.pack(fill='x', pady=3)

        ttk.Label(ksize_frame, text="Sobel Aperture:").pack(side='left')

        ksize_values = ["3", "5", "7"]
        self.ksize_combo = ttk.Combobox(
            ksize_frame,
            values=ksize_values,
            state='readonly',
            width=5
        )
        self.ksize_combo.current(0)
        self.ksize_combo.pack(side='left', padx=5)
        self.ksize_combo.bind('<<ComboboxSelected>>', self._on_ksize_change)

        # Harris k parameter
        k_frame = ttk.Frame(right_column)
        k_frame.pack(fill='x', pady=3)

        ttk.Label(k_frame, text="Harris k:").pack(side='left')

        k_slider = ttk.Scale(
            k_frame,
            from_=0.01,
            to=0.10,
            orient='horizontal',
            variable=self.k,
            command=self._on_k_change
        )
        k_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.k_label = ttk.Label(k_frame, text="0.04")
        self.k_label.pack(side='left', padx=5)

        # Threshold
        thresh_frame = ttk.Frame(right_column)
        thresh_frame.pack(fill='x', pady=3)

        ttk.Label(thresh_frame, text="Threshold:").pack(side='left')

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=0.001,
            to=0.1,
            orient='horizontal',
            variable=self.threshold,
            command=self._on_thresh_change
        )
        thresh_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh_label = ttk.Label(thresh_frame, text="0.010")
        self.thresh_label.pack(side='left', padx=5)

        # Corner color
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Corner Color (BGR):").pack(side='left')

        ttk.Label(color_frame, text="B:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_b).pack(side='left')

        ttk.Label(color_frame, text="G:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_g).pack(side='left')

        ttk.Label(color_frame, text="R:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.corner_color_r).pack(side='left')

        # Marker size
        marker_frame = ttk.Frame(right_column)
        marker_frame.pack(fill='x', pady=3)

        ttk.Label(marker_frame, text="Marker Size:").pack(side='left')

        marker_slider = ttk.Scale(
            marker_frame,
            from_=1,
            to=15,
            orient='horizontal',
            variable=self.marker_size,
            command=self._on_marker_change
        )
        marker_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.marker_label = ttk.Label(marker_frame, text="5")
        self.marker_label.pack(side='left', padx=5)

        # Corners found display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Corners found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_block_change(self, value):
        self.block_label.config(text=str(int(float(value))))

    def _on_ksize_change(self, event):
        self.ksize.set(int(self.ksize_combo.get()))

    def _on_k_change(self, value):
        self.k_label.config(text=f"{float(value):.2f}")

    def _on_thresh_change(self, value):
        self.thresh_label.config(text=f"{float(value):.3f}")

    def _on_marker_change(self, value):
        self.marker_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and mark corners on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = np.float32(gray)

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        block_size = self.block_size.get()
        ksize = self.ksize.get()
        k = self.k.get()
        threshold = self.threshold.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.corner_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.corner_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.corner_color_r.get()
        except Exception:
            color_r = 255
        color = (color_b, color_g, color_r)

        marker_size = self.marker_size.get()

        # Apply Harris corner detection
        harris = cv2.cornerHarris(gray, block_size, ksize, k)

        # Dilate to mark corners
        harris = cv2.dilate(harris, None)

        # Threshold and find corners
        corner_threshold = threshold * harris.max()
        corners = np.where(harris > corner_threshold)

        # Store raw values for pipeline use
        self.detected_corners = corners

        corners_count = len(corners[0])

        # Draw corners
        if self.draw_features.get():
            for y, x in zip(corners[0], corners[1]):
                cv2.circle(result, (x, y), marker_size, color, -1)

        # Update count
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(corners_count))

        return result

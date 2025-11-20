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
        self.return_mode_index = tk.IntVar(value=2)  # 0=dx, 1=dy, 2=Combined, 3=Magnitude, 4=Orientation, 5=Mask
        self.weight_x = tk.DoubleVar(value=0.5)
        self.weight_y = tk.DoubleVar(value=0.5)
        self.min_angle = tk.DoubleVar(value=0.0)
        self.max_angle = tk.DoubleVar(value=180.0)

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

        # Header section (skip if in pipeline - LabelFrame already shows name)
        if not getattr(self, '_in_pipeline', False):
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

        # dx control - dropdown
        dx_frame = ttk.Frame(right_column)
        dx_frame.pack(fill='x', pady=3)

        ttk.Label(dx_frame, text="dx (x derivative):").pack(side='left')

        self.dx_combo = ttk.Combobox(
            dx_frame,
            values=[0, 1, 2],
            state='readonly',
            width=5
        )
        self.dx_combo.current(1)  # Default to 1
        self.dx_combo.pack(side='left', padx=5)
        self.dx_combo.bind('<<ComboboxSelected>>', self._on_dx_change)

        # dy control - dropdown
        dy_frame = ttk.Frame(right_column)
        dy_frame.pack(fill='x', pady=3)

        ttk.Label(dy_frame, text="dy (y derivative):").pack(side='left')

        self.dy_combo = ttk.Combobox(
            dy_frame,
            values=[0, 1, 2],
            state='readonly',
            width=5
        )
        self.dy_combo.current(0)  # Default to 0
        self.dy_combo.pack(side='left', padx=5)
        self.dy_combo.bind('<<ComboboxSelected>>', self._on_dy_change)

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

        # Return mode - dropdown with descriptions
        return_frame = ttk.Frame(right_column)
        return_frame.pack(fill='x', pady=3)

        ttk.Label(return_frame, text="Return:", font=('TkDefaultFont', 12, 'bold')).pack(side='left')

        # Dropdown options with help text included
        return_options = [
            "gX (uses dx)",
            "gY (uses dy)",
            "Combined (cv2.addWeighted)",
            "Magnitude (√(gX² + gY²))",
            "Orientation (arctan2 → 0-180°)",
            "Mask (angle range filter)",
        ]

        self.return_combo = ttk.Combobox(
            return_frame,
            values=return_options,
            state='readonly',
            width=30
        )
        self.return_combo.current(2)  # Default to Combined
        self.return_combo.pack(side='left', padx=5)
        self.return_combo.bind('<<ComboboxSelected>>', self._on_return_change)

        # Weight X slider (for combined mode)
        weight_x_frame = ttk.Frame(right_column)
        weight_x_frame.pack(fill='x', pady=3)

        ttk.Label(weight_x_frame, text="Combined Weight X:").pack(side='left')

        weight_x_slider = ttk.Scale(
            weight_x_frame,
            from_=0,
            to=1,
            orient='horizontal',
            variable=self.weight_x,
            command=self._on_weight_x_change
        )
        weight_x_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.weight_x_label = ttk.Label(weight_x_frame, text="0.5")
        self.weight_x_label.pack(side='left', padx=5)

        # Weight Y slider (for combined mode)
        weight_y_frame = ttk.Frame(right_column)
        weight_y_frame.pack(fill='x', pady=3)

        ttk.Label(weight_y_frame, text="Combined Weight Y:").pack(side='left')

        weight_y_slider = ttk.Scale(
            weight_y_frame,
            from_=0,
            to=1,
            orient='horizontal',
            variable=self.weight_y,
            command=self._on_weight_y_change
        )
        weight_y_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.weight_y_label = ttk.Label(weight_y_frame, text="0.5")
        self.weight_y_label.pack(side='left', padx=5)

        # Min Angle slider (for mask mode)
        min_angle_frame = ttk.Frame(right_column)
        min_angle_frame.pack(fill='x', pady=3)

        ttk.Label(min_angle_frame, text="Mask Min Angle:").pack(side='left')

        min_angle_slider = ttk.Scale(
            min_angle_frame,
            from_=0,
            to=180,
            orient='horizontal',
            variable=self.min_angle,
            command=self._on_min_angle_change
        )
        min_angle_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.min_angle_label = ttk.Label(min_angle_frame, text="0°")
        self.min_angle_label.pack(side='left', padx=5)

        # Max Angle slider (for mask mode)
        max_angle_frame = ttk.Frame(right_column)
        max_angle_frame.pack(fill='x', pady=3)

        ttk.Label(max_angle_frame, text="Mask Max Angle:").pack(side='left')

        max_angle_slider = ttk.Scale(
            max_angle_frame,
            from_=0,
            to=180,
            orient='horizontal',
            variable=self.max_angle,
            command=self._on_max_angle_change
        )
        max_angle_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.max_angle_label = ttk.Label(max_angle_frame, text="180°")
        self.max_angle_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_depth_change(self, event):
        """Handle depth change"""
        self.depth_index.set(self.depth_combo.current())

    def _on_dx_change(self, event):
        """Handle dx change"""
        self.dx.set(int(self.dx_combo.get()))

    def _on_dy_change(self, event):
        """Handle dy change"""
        self.dy.set(int(self.dy_combo.get()))

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

    def _on_weight_x_change(self, value):
        """Handle weight X slider change"""
        weight = float(value)
        self.weight_x_label.config(text=f"{weight:.2f}")

    def _on_weight_y_change(self, value):
        """Handle weight Y slider change"""
        weight = float(value)
        self.weight_y_label.config(text=f"{weight:.2f}")

    def _on_min_angle_change(self, value):
        """Handle min angle slider change"""
        angle = float(value)
        self.min_angle_label.config(text=f"{angle:.0f}°")

    def _on_max_angle_change(self, value):
        """Handle max angle slider change"""
        angle = float(value)
        self.max_angle_label.config(text=f"{angle:.0f}°")

    def _on_return_change(self, event):
        """Handle return mode change"""
        self.return_mode_index.set(self.return_combo.current())

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

        # Get return mode: 0=gX, 1=gY, 2=Combined, 3=Magnitude, 4=Orientation
        return_mode = self.return_mode_index.get()

        if return_mode == 0:
            # gX - use dx value, force dy=0
            if dx == 0:
                dx = 1
            gradient = cv2.Sobel(gray, ddepth=ddepth, dx=dx, dy=0, ksize=ksize, scale=scale, delta=delta)

            # Convert to displayable format
            if ddepth in [cv2.CV_64F, cv2.CV_32F, cv2.CV_16S]:
                result = cv2.convertScaleAbs(gradient)
            else:
                result = gradient

        elif return_mode == 1:
            # gY - use dy value, force dx=0
            if dy == 0:
                dy = 1
            gradient = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=dy, ksize=ksize, scale=scale, delta=delta)

            # Convert to displayable format
            if ddepth in [cv2.CV_64F, cv2.CV_32F, cv2.CV_16S]:
                result = cv2.convertScaleAbs(gradient)
            else:
                result = gradient

        elif return_mode == 2:
            # Combined - weighted average of gX and gY
            gX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Convert to absolute values for combining
            absX = cv2.convertScaleAbs(gX)
            absY = cv2.convertScaleAbs(gY)

            # Combine the sobel X and Y representations into a single image
            weight_x = self.weight_x.get()
            weight_y = self.weight_y.get()
            result = cv2.addWeighted(absX, weight_x, absY, weight_y, 0)

        elif return_mode == 3:
            # Magnitude - √(gX² + gY²)
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute magnitude
            mag = np.sqrt((gX ** 2) + (gY ** 2))

            # Normalize to 0-255
            result = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)

        elif return_mode == 4:
            # Orientation - arctan2(gY, gX) mapped to 0-180°
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute orientation in degrees (0-180)
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            # Normalize to 0-255 for display
            result = (orientation * 255 / 180).astype(np.uint8)

        else:  # return_mode == 5
            # Mask - filter by angle range
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute orientation in degrees (0-180)
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            # Get angle range
            lower_angle = self.min_angle.get()
            upper_angle = self.max_angle.get()

            # Find all pixels within the angle boundaries
            idxs = np.where(orientation >= lower_angle, orientation, -1)
            idxs = np.where(orientation <= upper_angle, idxs, -1)
            result = np.zeros(gray.shape, dtype=np.uint8)
            result[idxs > -1] = 255

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

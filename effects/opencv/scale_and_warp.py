"""
Geometric transform effect using OpenCV.

Applies translation, rotation, and scaling transformations.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ScaleAndWarpEffect(BaseUIEffect):
    """Apply geometric transformations: translate, rotate, scale"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Translation
        self.translate_x = tk.IntVar(value=0)
        self.translate_y = tk.IntVar(value=0)

        # Rotation (degrees)
        self.rotation = tk.DoubleVar(value=0.0)

        # Scale
        self.scale = tk.DoubleVar(value=1.0)

        # Center point option
        self.use_image_center = tk.BooleanVar(value=True)
        self.center_x = tk.IntVar(value=width // 2)
        self.center_y = tk.IntVar(value=height // 2)

        # Border mode
        self.border_mode = tk.IntVar(value=0)  # Index into BORDER_MODES

    # Border modes for areas outside the image
    BORDER_MODES = [
        (cv2.BORDER_CONSTANT, "Constant (black)"),
        (cv2.BORDER_REPLICATE, "Replicate edge"),
        (cv2.BORDER_REFLECT, "Reflect"),
        (cv2.BORDER_WRAP, "Wrap"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "Warp Affine"

    @classmethod
    def get_description(cls) -> str:
        return "Apply translation, rotation, and scaling via affine transform"

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
                text="Geometric Transform",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.warpAffine(src, Matrix, dsize)",
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

        # Translation X
        tx_frame = ttk.Frame(right_column)
        tx_frame.pack(fill='x', pady=3)

        ttk.Label(tx_frame, text="Translate X:").pack(side='left')

        tx_slider = ttk.Scale(
            tx_frame,
            from_=-500,
            to=500,
            orient='horizontal',
            variable=self.translate_x,
            command=self._on_tx_change
        )
        tx_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.tx_label = ttk.Label(tx_frame, text="0")
        self.tx_label.pack(side='left', padx=5)

        # Translation Y
        ty_frame = ttk.Frame(right_column)
        ty_frame.pack(fill='x', pady=3)

        ttk.Label(ty_frame, text="Translate Y:").pack(side='left')

        ty_slider = ttk.Scale(
            ty_frame,
            from_=-500,
            to=500,
            orient='horizontal',
            variable=self.translate_y,
            command=self._on_ty_change
        )
        ty_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ty_label = ttk.Label(ty_frame, text="0")
        self.ty_label.pack(side='left', padx=5)

        # Rotation
        rot_frame = ttk.Frame(right_column)
        rot_frame.pack(fill='x', pady=3)

        ttk.Label(rot_frame, text="Rotation:").pack(side='left')

        rot_slider = ttk.Scale(
            rot_frame,
            from_=-180,
            to=180,
            orient='horizontal',
            variable=self.rotation,
            command=self._on_rotation_change
        )
        rot_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.rot_label = ttk.Label(rot_frame, text="0.0°")
        self.rot_label.pack(side='left', padx=5)

        # Scale
        scale_frame = ttk.Frame(right_column)
        scale_frame.pack(fill='x', pady=3)

        ttk.Label(scale_frame, text="Scale:").pack(side='left')

        scale_slider = ttk.Scale(
            scale_frame,
            from_=0.1,
            to=4.0,
            orient='horizontal',
            variable=self.scale,
            command=self._on_scale_change
        )
        scale_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.scale_label = ttk.Label(scale_frame, text="1.00x")
        self.scale_label.pack(side='left', padx=5)

        # Center point option - radio buttons
        center_frame = ttk.Frame(right_column)
        center_frame.pack(fill='x', pady=3)

        ttk.Label(center_frame, text="Center:").pack(side='left')

        ttk.Radiobutton(
            center_frame,
            text="Image Center",
            variable=self.use_image_center,
            value=True,
            command=self._on_center_toggle
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            center_frame,
            text="Custom X/Y",
            variable=self.use_image_center,
            value=False,
            command=self._on_center_toggle
        ).pack(side='left', padx=5)

        # Custom center controls (always visible, but spinboxes disabled when using image center)
        custom_center_frame = ttk.Frame(right_column)
        custom_center_frame.pack(fill='x', pady=3)

        ttk.Label(custom_center_frame, text="Center X:").pack(side='left', padx=(20, 0))
        self.cx_spin = ttk.Spinbox(
            custom_center_frame,
            from_=0,
            to=self.width,
            width=5,
            textvariable=self.center_x
        )
        self.cx_spin.pack(side='left', padx=(2, 10))

        ttk.Label(custom_center_frame, text="Center Y:").pack(side='left')
        self.cy_spin = ttk.Spinbox(
            custom_center_frame,
            from_=0,
            to=self.height,
            width=5,
            textvariable=self.center_y
        )
        self.cy_spin.pack(side='left', padx=2)

        # Initially disable custom center spinboxes if using image center
        if self.use_image_center.get():
            self.cx_spin.config(state='disabled')
            self.cy_spin.config(state='disabled')

        # Border mode
        border_frame = ttk.Frame(right_column)
        border_frame.pack(fill='x', pady=3)

        ttk.Label(border_frame, text="Border Mode:").pack(side='left')

        border_values = [name for _, name in self.BORDER_MODES]
        self.border_combo = ttk.Combobox(
            border_frame,
            values=border_values,
            state='readonly',
            width=18
        )
        self.border_combo.current(0)
        self.border_combo.pack(side='left', padx=5)
        self.border_combo.bind('<<ComboboxSelected>>', self._on_border_change)

        return self.control_panel

    def _on_tx_change(self, value):
        """Handle translate X slider change"""
        self.tx_label.config(text=str(int(float(value))))

    def _on_ty_change(self, value):
        """Handle translate Y slider change"""
        self.ty_label.config(text=str(int(float(value))))

    def _on_rotation_change(self, value):
        """Handle rotation slider change"""
        self.rot_label.config(text=f"{float(value):.1f}°")

    def _on_scale_change(self, value):
        """Handle scale slider change"""
        self.scale_label.config(text=f"{float(value):.2f}x")

    def _on_center_toggle(self):
        """Handle center point toggle"""
        if self.use_image_center.get():
            self.cx_spin.config(state='disabled')
            self.cy_spin.config(state='disabled')
        else:
            self.cx_spin.config(state='normal')
            self.cy_spin.config(state='normal')

    def _on_border_change(self, event):
        """Handle border mode change"""
        self.border_mode.set(self.border_combo.current())

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply geometric transformation to the frame"""
        if not self.enabled.get():
            return frame

        height, width = frame.shape[:2]

        # Get parameters
        tx = self.translate_x.get()
        ty = self.translate_y.get()
        angle = self.rotation.get()
        scale = self.scale.get()

        # Determine center point
        if self.use_image_center.get():
            cx, cy = width / 2, height / 2
        else:
            try:
                cx = int(self.center_x.get())
                cy = int(self.center_y.get())
            except (ValueError, tk.TclError):
                cx, cy = width / 2, height / 2

        # Build transformation matrix
        # getRotationMatrix2D handles rotation and scale around center
        # Negate angle so positive = clockwise (more intuitive)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, scale)

        # Add translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Get border mode
        border_idx = self.border_mode.get()
        border_mode = self.BORDER_MODES[border_idx][0]

        # Apply transformation
        result = cv2.warpAffine(
            frame,
            M,
            (width, height),
            borderMode=border_mode
        )

        return result

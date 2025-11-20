"""
SIFT (Scale-Invariant Feature Transform) feature detection using OpenCV.

Detects and visualizes keypoints that are robust to scale, rotation, and illumination changes.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class SIFTEffect(BaseUIEffect):
    """Detect and visualize SIFT keypoints"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.n_features = tk.IntVar(value=500)
        self.n_octave_layers = tk.IntVar(value=3)
        self.contrast_threshold = tk.DoubleVar(value=0.04)
        self.edge_threshold = tk.DoubleVar(value=10)
        self.sigma = tk.DoubleVar(value=1.6)
        self.show_rich_keypoints = tk.BooleanVar(value=True)
        self.keypoint_color = tk.StringVar(value="green")

    @classmethod
    def get_name(cls) -> str:
        return "SIFT: Scale-Invariant Features"

    @classmethod
    def get_description(cls) -> str:
        return "Scale and rotation invariant keypoint detection"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def create_control_panel(self, parent):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section (skip if in pipeline)
        if not getattr(self, '_in_pipeline', False):
            header_frame = ttk.Frame(self.control_panel)
            header_frame.pack(fill='x', **padding)

            title_label = ttk.Label(
                header_frame,
                text="SIFT: Scale-Invariant Feature Transform (accurate)",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            desc_label = ttk.Label(
                header_frame,
                text="cv2.SIFT_create(nfeatures, nOctaveLayers, ...)",
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

        # Number of features
        nf_frame = ttk.Frame(right_column)
        nf_frame.pack(fill='x', pady=3)

        ttk.Label(nf_frame, text="Max Features:").pack(side='left')

        def on_nf_change(*args):
            self.nf_label.config(text=str(self.n_features.get()))

        nf_slider = ttk.Scale(
            nf_frame,
            from_=10,
            to=2000,
            orient='horizontal',
            variable=self.n_features,
            command=lambda v: on_nf_change()
        )
        nf_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.nf_label = ttk.Label(nf_frame, text="500", width=5)
        self.nf_label.pack(side='left', padx=5)

        self.n_features.trace_add("write", on_nf_change)

        # Contrast threshold
        ct_frame = ttk.Frame(right_column)
        ct_frame.pack(fill='x', pady=3)

        ttk.Label(ct_frame, text="Contrast Thresh:").pack(side='left')

        def on_ct_change(*args):
            self.ct_label.config(text=f"{self.contrast_threshold.get():.3f}")

        ct_slider = ttk.Scale(
            ct_frame,
            from_=0.01,
            to=0.2,
            orient='horizontal',
            variable=self.contrast_threshold,
            command=lambda v: on_ct_change()
        )
        ct_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ct_label = ttk.Label(ct_frame, text="0.040", width=6)
        self.ct_label.pack(side='left', padx=5)

        self.contrast_threshold.trace_add("write", on_ct_change)

        # Edge threshold
        et_frame = ttk.Frame(right_column)
        et_frame.pack(fill='x', pady=3)

        ttk.Label(et_frame, text="Edge Thresh:").pack(side='left')

        def on_et_change(*args):
            self.et_label.config(text=f"{self.edge_threshold.get():.1f}")

        et_slider = ttk.Scale(
            et_frame,
            from_=1,
            to=50,
            orient='horizontal',
            variable=self.edge_threshold,
            command=lambda v: on_et_change()
        )
        et_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.et_label = ttk.Label(et_frame, text="10.0", width=5)
        self.et_label.pack(side='left', padx=5)

        self.edge_threshold.trace_add("write", on_et_change)

        # Rich keypoints checkbox
        rich_frame = ttk.Frame(right_column)
        rich_frame.pack(fill='x', pady=3)

        ttk.Checkbutton(
            rich_frame,
            text="Show size & orientation",
            variable=self.show_rich_keypoints
        ).pack(side='left')

        # Color selection
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Color:").pack(side='left')

        for color in ["green", "red", "blue", "yellow", "white"]:
            ttk.Radiobutton(
                color_frame,
                text=color.capitalize(),
                variable=self.keypoint_color,
                value=color
            ).pack(side='left', padx=3)

        return self.control_panel

    def _get_color_bgr(self):
        """Get BGR color tuple from color name"""
        colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255)
        }
        return colors.get(self.keypoint_color.get(), (0, 255, 0))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw SIFT keypoints"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create SIFT detector
        sift = cv2.SIFT_create(
            nfeatures=self.n_features.get(),
            nOctaveLayers=self.n_octave_layers.get(),
            contrastThreshold=self.contrast_threshold.get(),
            edgeThreshold=self.edge_threshold.get(),
            sigma=self.sigma.get()
        )

        # Detect keypoints
        keypoints = sift.detect(gray, None)

        # Draw keypoints
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if self.show_rich_keypoints.get() else 0
        result = cv2.drawKeypoints(
            frame,
            keypoints,
            None,
            color=self._get_color_bgr(),
            flags=flags
        )

        return result

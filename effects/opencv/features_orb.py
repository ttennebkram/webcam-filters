"""
ORB (Oriented FAST and Rotated BRIEF) feature detection using OpenCV.

Fast, free alternative to SIFT/SURF. Great for real-time applications.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ORBEffect(BaseUIEffect):
    """Detect and visualize ORB keypoints"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.n_features = tk.IntVar(value=500)
        self.scale_factor = tk.DoubleVar(value=1.2)
        self.n_levels = tk.IntVar(value=8)
        self.edge_threshold = tk.IntVar(value=31)
        self.first_level = tk.IntVar(value=0)
        self.wta_k = tk.IntVar(value=2)
        self.patch_size = tk.IntVar(value=31)
        self.fast_threshold = tk.IntVar(value=20)
        self.show_rich_keypoints = tk.BooleanVar(value=True)
        self.keypoint_color = tk.StringVar(value="green")

    @classmethod
    def get_name(cls) -> str:
        return "ORB: Fast Real-time Features"

    @classmethod
    def get_description(cls) -> str:
        return "Fast, free keypoint detection for real-time use"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.ORB_create(nfeatures, scaleFactor, nlevels, ...)"

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
                text="ORB: Oriented FAST and Rotated BRIEF (very fast)",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            desc_label = ttk.Label(
                header_frame,
                text="cv2.ORB_create(nfeatures, scaleFactor, nlevels, ...)",
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
            to=5000,
            orient='horizontal',
            variable=self.n_features,
            command=lambda v: on_nf_change()
        )
        nf_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.nf_label = ttk.Label(nf_frame, text="500", width=5)
        self.nf_label.pack(side='left', padx=5)

        self.n_features.trace_add("write", on_nf_change)

        # Scale factor
        sf_frame = ttk.Frame(right_column)
        sf_frame.pack(fill='x', pady=3)

        ttk.Label(sf_frame, text="Scale Factor:").pack(side='left')

        def on_sf_change(*args):
            self.sf_label.config(text=f"{self.scale_factor.get():.2f}")

        sf_slider = ttk.Scale(
            sf_frame,
            from_=1.01,
            to=2.0,
            orient='horizontal',
            variable=self.scale_factor,
            command=lambda v: on_sf_change()
        )
        sf_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.sf_label = ttk.Label(sf_frame, text="1.20", width=5)
        self.sf_label.pack(side='left', padx=5)

        self.scale_factor.trace_add("write", on_sf_change)

        # Number of pyramid levels
        nl_frame = ttk.Frame(right_column)
        nl_frame.pack(fill='x', pady=3)

        ttk.Label(nl_frame, text="Pyramid Levels:").pack(side='left')

        def on_nl_change(*args):
            self.nl_label.config(text=str(self.n_levels.get()))

        nl_slider = ttk.Scale(
            nl_frame,
            from_=1,
            to=16,
            orient='horizontal',
            variable=self.n_levels,
            command=lambda v: on_nl_change()
        )
        nl_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.nl_label = ttk.Label(nl_frame, text="8", width=3)
        self.nl_label.pack(side='left', padx=5)

        self.n_levels.trace_add("write", on_nl_change)

        # FAST threshold
        ft_frame = ttk.Frame(right_column)
        ft_frame.pack(fill='x', pady=3)

        ttk.Label(ft_frame, text="FAST Thresh:").pack(side='left')

        def on_ft_change(*args):
            self.ft_label.config(text=str(self.fast_threshold.get()))

        ft_slider = ttk.Scale(
            ft_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.fast_threshold,
            command=lambda v: on_ft_change()
        )
        ft_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.ft_label = ttk.Label(ft_frame, text="20", width=4)
        self.ft_label.pack(side='left', padx=5)

        self.fast_threshold.trace_add("write", on_ft_change)

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
        """Detect and draw ORB keypoints"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create ORB detector
        orb = cv2.ORB_create(
            nfeatures=self.n_features.get(),
            scaleFactor=self.scale_factor.get(),
            nlevels=self.n_levels.get(),
            edgeThreshold=self.edge_threshold.get(),
            firstLevel=self.first_level.get(),
            WTA_K=self.wta_k.get(),
            patchSize=self.patch_size.get(),
            fastThreshold=self.fast_threshold.get()
        )

        # Detect keypoints
        keypoints = orb.detect(gray, None)

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

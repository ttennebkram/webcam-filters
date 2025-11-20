"""
Contour detection and drawing effect using OpenCV.

Finds contours in images and draws them with various options.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ContoursEffect(BaseUIEffect):
    """Find and draw contours on images"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Background option: show original image or black
        self.show_original = tk.BooleanVar(value=True)

        # Contour retrieval mode
        self.retrieval_mode = tk.IntVar(value=0)  # Index into RETRIEVAL_MODES

        # Contour approximation method
        self.approx_method = tk.IntVar(value=0)  # Index into APPROX_METHODS

        # Drawing options
        self.contour_color_r = tk.IntVar(value=0)
        self.contour_color_g = tk.IntVar(value=255)
        self.contour_color_b = tk.IntVar(value=0)
        self.thickness = tk.IntVar(value=2)

        # Filter options
        self.min_area = tk.IntVar(value=100)
        self.max_area = tk.IntVar(value=100000)

        # Pre-processing: threshold before finding contours
        self.threshold_value = tk.IntVar(value=127)

    # Contour retrieval modes
    RETRIEVAL_MODES = [
        (cv2.RETR_EXTERNAL, "RETR_EXTERNAL", "Only outermost contours"),
        (cv2.RETR_LIST, "RETR_LIST", "All contours, no hierarchy"),
        (cv2.RETR_CCOMP, "RETR_CCOMP", "Two-level hierarchy"),
        (cv2.RETR_TREE, "RETR_TREE", "Full hierarchy tree"),
    ]

    # Contour approximation methods
    APPROX_METHODS = [
        (cv2.CHAIN_APPROX_NONE, "CHAIN_APPROX_NONE", "All points"),
        (cv2.CHAIN_APPROX_SIMPLE, "CHAIN_APPROX_SIMPLE", "Compress segments"),
        (cv2.CHAIN_APPROX_TC89_L1, "CHAIN_APPROX_TC89_L1", "Teh-Chin L1"),
        (cv2.CHAIN_APPROX_TC89_KCOS, "CHAIN_APPROX_TC89_KCOS", "Teh-Chin KCOS"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "Contours"

    @classmethod
    def get_description(cls) -> str:
        return "Find and draw contours on images"

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
            text="Contour Detection",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(anchor='w')

        # Method signature
        signature_label = ttk.Label(
            header_frame,
            text="cv2.findContours(image, mode, method)",
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

        # Background option
        bg_frame = ttk.Frame(right_column)
        bg_frame.pack(fill='x', pady=3)

        ttk.Label(bg_frame, text="Background:").pack(side='left')

        ttk.Radiobutton(
            bg_frame,
            text="Original Image",
            variable=self.show_original,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            bg_frame,
            text="Black",
            variable=self.show_original,
            value=False
        ).pack(side='left', padx=5)

        # Threshold control (for pre-processing)
        thresh_frame = ttk.Frame(right_column)
        thresh_frame.pack(fill='x', pady=3)

        ttk.Label(thresh_frame, text="Threshold:").pack(side='left')

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.threshold_value,
            command=self._on_thresh_change
        )
        thresh_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh_label = ttk.Label(thresh_frame, text="127")
        self.thresh_label.pack(side='left', padx=5)

        # Retrieval mode dropdown
        retrieval_frame = ttk.Frame(right_column)
        retrieval_frame.pack(fill='x', pady=3)

        ttk.Label(retrieval_frame, text="Retrieval Mode:").pack(side='left')

        retrieval_values = [f"{name}" for _, name, _ in self.RETRIEVAL_MODES]
        self.retrieval_combo = ttk.Combobox(
            retrieval_frame,
            values=retrieval_values,
            state='readonly',
            width=20
        )
        self.retrieval_combo.current(0)
        self.retrieval_combo.pack(side='left', padx=5)
        self.retrieval_combo.bind('<<ComboboxSelected>>', self._on_retrieval_change)

        # Retrieval mode description
        self.retrieval_desc = ttk.Label(
            retrieval_frame,
            text=self.RETRIEVAL_MODES[0][2],
            font=('TkDefaultFont', 10, 'italic')
        )
        self.retrieval_desc.pack(side='left', padx=5)

        # Approximation method dropdown
        approx_frame = ttk.Frame(right_column)
        approx_frame.pack(fill='x', pady=3)

        ttk.Label(approx_frame, text="Approximation:").pack(side='left')

        approx_values = [f"{name}" for _, name, _ in self.APPROX_METHODS]
        self.approx_combo = ttk.Combobox(
            approx_frame,
            values=approx_values,
            state='readonly',
            width=25
        )
        self.approx_combo.current(0)
        self.approx_combo.pack(side='left', padx=5)
        self.approx_combo.bind('<<ComboboxSelected>>', self._on_approx_change)

        # Thickness control
        thick_frame = ttk.Frame(right_column)
        thick_frame.pack(fill='x', pady=3)

        ttk.Label(thick_frame, text="Line Thickness:").pack(side='left')

        thick_slider = ttk.Scale(
            thick_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.thickness,
            command=self._on_thickness_change
        )
        thick_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thick_label = ttk.Label(thick_frame, text="2")
        self.thick_label.pack(side='left', padx=5)

        # Color controls
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Contour Color (BGR):").pack(side='left')

        # B
        ttk.Label(color_frame, text="B:").pack(side='left', padx=(10, 2))
        b_spin = ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.contour_color_b)
        b_spin.pack(side='left')

        # G
        ttk.Label(color_frame, text="G:").pack(side='left', padx=(10, 2))
        g_spin = ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.contour_color_g)
        g_spin.pack(side='left')

        # R
        ttk.Label(color_frame, text="R:").pack(side='left', padx=(10, 2))
        r_spin = ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.contour_color_r)
        r_spin.pack(side='left')

        # Area filter controls
        area_frame = ttk.Frame(right_column)
        area_frame.pack(fill='x', pady=3)

        ttk.Label(area_frame, text="Min Area:").pack(side='left')

        min_area_slider = ttk.Scale(
            area_frame,
            from_=0,
            to=10000,
            orient='horizontal',
            variable=self.min_area,
            command=self._on_min_area_change
        )
        min_area_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.min_area_label = ttk.Label(area_frame, text="100")
        self.min_area_label.pack(side='left', padx=5)

        # Max area
        max_frame = ttk.Frame(right_column)
        max_frame.pack(fill='x', pady=3)

        ttk.Label(max_frame, text="Max Area:").pack(side='left')

        max_area_slider = ttk.Scale(
            max_frame,
            from_=1000,
            to=500000,
            orient='horizontal',
            variable=self.max_area,
            command=self._on_max_area_change
        )
        max_area_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.max_area_label = ttk.Label(max_frame, text="100000")
        self.max_area_label.pack(side='left', padx=5)

        # Contour count display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Contours found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_thresh_change(self, value):
        """Handle threshold slider change"""
        self.thresh_label.config(text=str(int(float(value))))

    def _on_retrieval_change(self, event):
        """Handle retrieval mode change"""
        idx = self.retrieval_combo.current()
        self.retrieval_mode.set(idx)
        self.retrieval_desc.config(text=self.RETRIEVAL_MODES[idx][2])

    def _on_approx_change(self, event):
        """Handle approximation method change"""
        self.approx_method.set(self.approx_combo.current())

    def _on_thickness_change(self, value):
        """Handle thickness slider change"""
        self.thick_label.config(text=str(int(float(value))))

    def _on_min_area_change(self, value):
        """Handle min area slider change"""
        self.min_area_label.config(text=str(int(float(value))))

    def _on_max_area_change(self, value):
        """Handle max area slider change"""
        self.max_area_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Find and draw contours on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply threshold to get binary image
        thresh_val = self.threshold_value.get()
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Get contour parameters
        retrieval_idx = self.retrieval_mode.get()
        approx_idx = self.approx_method.get()

        retrieval_mode = self.RETRIEVAL_MODES[retrieval_idx][0]
        approx_method = self.APPROX_METHODS[approx_idx][0]

        # Find contours
        contours, hierarchy = cv2.findContours(binary, retrieval_mode, approx_method)

        # Filter contours by area
        min_area = self.min_area.get()
        max_area = self.max_area.get()

        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered_contours.append(contour)

        # Update count display
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(len(filtered_contours)))

        # Create output image
        if self.show_original.get():
            # Draw on original image
            result = frame.copy()
        else:
            # Draw on black background
            result = np.zeros_like(frame)

        # Get drawing parameters
        color = (
            self.contour_color_b.get(),
            self.contour_color_g.get(),
            self.contour_color_r.get()
        )
        thickness = self.thickness.get()

        # Draw contours
        cv2.drawContours(result, filtered_contours, -1, color, thickness)

        return result

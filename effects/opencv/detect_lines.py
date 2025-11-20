"""
Hough Line detection effect using OpenCV.

Detects straight lines in images using the Hough Transform.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class HoughLinesEffect(BaseUIEffect):
    """Detect and draw lines using Hough Transform"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Use probabilistic version
        self.use_probabilistic = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_lines = None

        # Common parameters
        self.rho = tk.DoubleVar(value=1.0)  # Distance resolution
        self.theta_divisions = tk.IntVar(value=180)  # Angle resolution (divisions of pi)
        self.threshold = tk.IntVar(value=50)  # Accumulator threshold

        # HoughLinesP specific
        self.min_line_length = tk.IntVar(value=50)
        self.max_line_gap = tk.IntVar(value=10)

        # Max lines to draw (prevents screen fill with standard HoughLines)
        self.max_lines = tk.IntVar(value=100)

        # Drawing options
        self.line_color_b = tk.IntVar(value=0)
        self.line_color_g = tk.IntVar(value=0)
        self.line_color_r = tk.IntVar(value=255)
        self.thickness = tk.IntVar(value=2)

        # Canny pre-processing
        self.canny_thresh1 = tk.IntVar(value=50)
        self.canny_thresh2 = tk.IntVar(value=150)

    @classmethod
    def get_name(cls) -> str:
        return "Hough Lines"

    @classmethod
    def get_description(cls) -> str:
        return "Detect straight lines using Hough Transform"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.HoughLinesP(image, rho, theta, threshold)"

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
                text="Hough Lines",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.HoughLinesP(edges, rho, theta, threshold)",
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

        # Method selection
        method_frame = ttk.Frame(right_column)
        method_frame.pack(fill='x', pady=3)

        ttk.Label(method_frame, text="Method:").pack(side='left')

        ttk.Radiobutton(
            method_frame,
            text="Probabilistic (HoughLinesP)",
            variable=self.use_probabilistic,
            value=True
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            method_frame,
            text="Standard (HoughLines)",
            variable=self.use_probabilistic,
            value=False
        ).pack(side='left', padx=5)

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

        # Canny thresholds for edge detection
        canny1_frame = ttk.Frame(right_column)
        canny1_frame.pack(fill='x', pady=3)

        ttk.Label(canny1_frame, text="Canny Thresh 1:").pack(side='left')

        canny1_slider = ttk.Scale(
            canny1_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.canny_thresh1,
            command=self._on_canny1_change
        )
        canny1_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.canny1_label = ttk.Label(canny1_frame, text="50")
        self.canny1_label.pack(side='left', padx=5)

        canny2_frame = ttk.Frame(right_column)
        canny2_frame.pack(fill='x', pady=3)

        ttk.Label(canny2_frame, text="Canny Thresh 2:").pack(side='left')

        canny2_slider = ttk.Scale(
            canny2_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.canny_thresh2,
            command=self._on_canny2_change
        )
        canny2_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.canny2_label = ttk.Label(canny2_frame, text="150")
        self.canny2_label.pack(side='left', padx=5)

        # Threshold (accumulator)
        thresh_frame = ttk.Frame(right_column)
        thresh_frame.pack(fill='x', pady=3)

        ttk.Label(thresh_frame, text="Threshold:").pack(side='left')

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=1,
            to=200,
            orient='horizontal',
            variable=self.threshold,
            command=self._on_thresh_change
        )
        thresh_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh_label = ttk.Label(thresh_frame, text="50")
        self.thresh_label.pack(side='left', padx=5)

        # Min line length (HoughLinesP)
        minlen_frame = ttk.Frame(right_column)
        minlen_frame.pack(fill='x', pady=3)

        ttk.Label(minlen_frame, text="Min Line Length:").pack(side='left')

        minlen_slider = ttk.Scale(
            minlen_frame,
            from_=1,
            to=200,
            orient='horizontal',
            variable=self.min_line_length,
            command=self._on_minlen_change
        )
        minlen_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.minlen_label = ttk.Label(minlen_frame, text="50")
        self.minlen_label.pack(side='left', padx=5)

        # Max line gap (HoughLinesP)
        maxgap_frame = ttk.Frame(right_column)
        maxgap_frame.pack(fill='x', pady=3)

        ttk.Label(maxgap_frame, text="Max Line Gap:").pack(side='left')

        maxgap_slider = ttk.Scale(
            maxgap_frame,
            from_=1,
            to=100,
            orient='horizontal',
            variable=self.max_line_gap,
            command=self._on_maxgap_change
        )
        maxgap_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.maxgap_label = ttk.Label(maxgap_frame, text="10")
        self.maxgap_label.pack(side='left', padx=5)

        # Max lines to draw
        maxlines_frame = ttk.Frame(right_column)
        maxlines_frame.pack(fill='x', pady=3)

        ttk.Label(maxlines_frame, text="Max Lines:").pack(side='left')

        maxlines_slider = ttk.Scale(
            maxlines_frame,
            from_=1,
            to=500,
            orient='horizontal',
            variable=self.max_lines,
            command=self._on_maxlines_change
        )
        maxlines_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.maxlines_label = ttk.Label(maxlines_frame, text="100")
        self.maxlines_label.pack(side='left', padx=5)

        # Line color
        color_frame = ttk.Frame(right_column)
        color_frame.pack(fill='x', pady=3)

        ttk.Label(color_frame, text="Line Color (BGR):").pack(side='left')

        ttk.Label(color_frame, text="B:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.line_color_b).pack(side='left')

        ttk.Label(color_frame, text="G:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.line_color_g).pack(side='left')

        ttk.Label(color_frame, text="R:").pack(side='left', padx=(10, 2))
        ttk.Spinbox(color_frame, from_=0, to=255, width=4, textvariable=self.line_color_r).pack(side='left')

        # Thickness
        thick_frame = ttk.Frame(right_column)
        thick_frame.pack(fill='x', pady=3)

        ttk.Label(thick_frame, text="Line Thickness:").pack(side='left')

        thick_slider = ttk.Scale(
            thick_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.thickness,
            command=self._on_thick_change
        )
        thick_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thick_label = ttk.Label(thick_frame, text="2")
        self.thick_label.pack(side='left', padx=5)

        # Lines found display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Lines found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_canny1_change(self, value):
        self.canny1_label.config(text=str(int(float(value))))

    def _on_canny2_change(self, value):
        self.canny2_label.config(text=str(int(float(value))))

    def _on_thresh_change(self, value):
        self.thresh_label.config(text=str(int(float(value))))

    def _on_minlen_change(self, value):
        self.minlen_label.config(text=str(int(float(value))))

    def _on_maxgap_change(self, value):
        self.maxgap_label.config(text=str(int(float(value))))

    def _on_maxlines_change(self, value):
        self.maxlines_label.config(text=str(int(float(value))))

    def _on_thick_change(self, value):
        self.thick_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw lines on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_thresh1.get(), self.canny_thresh2.get())

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        rho = self.rho.get()
        theta = np.pi / self.theta_divisions.get()
        threshold = self.threshold.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.line_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.line_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.line_color_r.get()
        except Exception:
            color_r = 255
        color = (color_b, color_g, color_r)

        thickness = self.thickness.get()
        max_lines = self.max_lines.get()

        lines_count = 0

        if self.use_probabilistic.get():
            # Probabilistic Hough Transform
            min_length = self.min_line_length.get()
            max_gap = self.max_line_gap.get()

            lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                                    minLineLength=min_length, maxLineGap=max_gap)

            # Store raw values for pipeline use
            self.detected_lines = lines

            if lines is not None:
                lines_count = len(lines)
                if self.draw_features.get():
                    for i, line in enumerate(lines):
                        if i >= max_lines:
                            break
                        x1, y1, x2, y2 = line[0]
                        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        else:
            # Standard Hough Transform
            lines = cv2.HoughLines(edges, rho, theta, threshold)

            # Store raw values for pipeline use
            self.detected_lines = lines

            if lines is not None:
                lines_count = len(lines)
                if self.draw_features.get():
                    for i, line in enumerate(lines):
                        if i >= max_lines:
                            break
                        rho_val, theta_val = line[0]
                        cos_t = np.cos(theta_val)
                        sin_t = np.sin(theta_val)
                        x0 = cos_t * rho_val
                        y0 = sin_t * rho_val
                        # Draw line extending across image
                        # Direction perpendicular to normal: (-sin, cos)
                        length = 2000
                        x1 = int(x0 - length * sin_t)
                        y1 = int(y0 + length * cos_t)
                        x2 = int(x0 + length * sin_t)
                        y2 = int(y0 - length * cos_t)
                        cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        # Update count
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(lines_count))

        return result

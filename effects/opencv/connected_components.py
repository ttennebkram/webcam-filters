"""
Connected components labeling effect using OpenCV.

Labels connected regions (blobs) in a binary image with different colors.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


class ConnectedComponentsEffect(BaseUIEffect):
    """Label connected components with different colors"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Threshold for binarization
        self.threshold = tk.IntVar(value=127)

        # Invert threshold
        self.invert = tk.BooleanVar(value=False)

        # Connectivity (4 or 8)
        self.connectivity = tk.IntVar(value=8)

        # Min area filter
        self.min_area = tk.IntVar(value=0)

        # Storage for labels
        self.num_labels = 0
        self.labels = None
        self.stats = None
        self.centroids = None

    @classmethod
    def get_name(cls) -> str:
        return "Connected Components"

    @classmethod
    def get_description(cls) -> str:
        return "Label connected regions with different colors"

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
                text="Connected Components",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Method signature
            signature_label = ttk.Label(
                header_frame,
                text="cv2.connectedComponentsWithStats(image, connectivity)",
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

        # Threshold
        thresh_frame = ttk.Frame(right_column)
        thresh_frame.pack(fill='x', pady=3)

        ttk.Label(thresh_frame, text="Threshold:").pack(side='left')

        thresh_slider = ttk.Scale(
            thresh_frame,
            from_=0,
            to=255,
            orient='horizontal',
            variable=self.threshold,
            command=self._on_thresh_change
        )
        thresh_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.thresh_label = ttk.Label(thresh_frame, text="127")
        self.thresh_label.pack(side='left', padx=5)

        # Invert checkbox
        invert_frame = ttk.Frame(right_column)
        invert_frame.pack(fill='x', pady=3)

        ttk.Checkbutton(
            invert_frame,
            text="Invert Threshold",
            variable=self.invert
        ).pack(side='left')

        # Connectivity
        conn_frame = ttk.Frame(right_column)
        conn_frame.pack(fill='x', pady=3)

        ttk.Label(conn_frame, text="Connectivity:").pack(side='left')

        ttk.Radiobutton(
            conn_frame,
            text="4",
            variable=self.connectivity,
            value=4
        ).pack(side='left', padx=(10, 5))

        ttk.Radiobutton(
            conn_frame,
            text="8",
            variable=self.connectivity,
            value=8
        ).pack(side='left', padx=5)

        # Min area
        minarea_frame = ttk.Frame(right_column)
        minarea_frame.pack(fill='x', pady=3)

        ttk.Label(minarea_frame, text="Min Area:").pack(side='left')

        minarea_slider = ttk.Scale(
            minarea_frame,
            from_=0,
            to=1000,
            orient='horizontal',
            variable=self.min_area,
            command=self._on_minarea_change
        )
        minarea_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.minarea_label = ttk.Label(minarea_frame, text="0")
        self.minarea_label.pack(side='left', padx=5)

        # Components found display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Components found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        return self.control_panel

    def _on_thresh_change(self, value):
        self.thresh_label.config(text=str(int(float(value))))

    def _on_minarea_change(self, value):
        self.minarea_label.config(text=str(int(float(value))))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Label connected components and colorize them"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply threshold
        threshold = self.threshold.get()
        if self.invert.get():
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Get connected components with stats
        connectivity = self.connectivity.get()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity, cv2.CV_32S
        )

        # Store for pipeline use
        self.num_labels = num_labels
        self.labels = labels
        self.stats = stats
        self.centroids = centroids

        # Create colored output
        # Generate random colors for each label (excluding background)
        np.random.seed(42)  # Consistent colors
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black

        # Filter by min area
        min_area = self.min_area.get()
        if min_area > 0:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    colors[i] = [0, 0, 0]  # Set small components to black

        # Map labels to colors
        result = colors[labels]

        # Count visible components
        visible_count = 0
        for i in range(1, num_labels):
            if not np.array_equal(colors[i], [0, 0, 0]):
                visible_count += 1

        # Update count
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(visible_count))

        return result

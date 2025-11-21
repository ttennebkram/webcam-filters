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

        # Pre-processing: threshold before finding contours
        self.threshold_value = tk.IntVar(value=127)

        # Sorting options
        self.sort_method = tk.IntVar(value=0)  # Index into SORT_METHODS

        # Contour index selection (after sorting)
        self.min_index = tk.IntVar(value=1)  # 1-based for user display
        self.max_index = tk.IntVar(value=100)

        # Drawing mode
        self.draw_mode = tk.IntVar(value=0)  # Index into DRAW_MODES

    # Drawing modes
    DRAW_MODES = [
        ("Contours", "contours"),
        ("Bounding Rectangles", "bounding_rect"),
        ("Rotated Bounding Rectangles", "rotated_rect"),
        ("Enclosing Circles", "enclosing_circle"),
        ("Fitted Ellipses", "fitted_ellipse"),
        ("Convex Hulls", "convex_hull"),
        ("Centroids", "centroids"),
    ]

    # Sorting methods
    SORT_METHODS = [
        ("None", None),
        ("Top-Bottom-Left-Right", "tb_lr"),
        ("Left-Right-Top-Bottom", "lr_tb"),
        ("Area (largest first)", "area_desc"),
        ("Area (smallest first)", "area_asc"),
        ("Perimeter (largest first)", "perim_desc"),
        ("Perimeter (smallest first)", "perim_asc"),
        ("Circularity (most circular)", "circ_desc"),
        ("Circularity (least circular)", "circ_asc"),
        ("Aspect Ratio (widest first)", "aspect_desc"),
        ("Aspect Ratio (tallest first)", "aspect_asc"),
        ("Extent (most filled first)", "extent_desc"),
        ("Extent (least filled first)", "extent_asc"),
        ("Solidity (most convex first)", "solid_desc"),
        ("Solidity (least convex first)", "solid_asc"),
    ]

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
    def get_method_signature(cls) -> str:
        return "cv2.findContours(image, mode, method)"

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

        # Sorting dropdown
        sort_frame = ttk.Frame(right_column)
        sort_frame.pack(fill='x', pady=3)

        ttk.Label(sort_frame, text="Sort By:").pack(side='left')

        sort_values = [name for name, _ in self.SORT_METHODS]
        self.sort_combo = ttk.Combobox(
            sort_frame,
            values=sort_values,
            state='readonly',
            width=28
        )
        self.sort_combo.current(0)
        self.sort_combo.pack(side='left', padx=5)
        self.sort_combo.bind('<<ComboboxSelected>>', self._on_sort_change)

        # Min/Max index selection (after sorting)
        index_frame = ttk.Frame(right_column)
        index_frame.pack(fill='x', pady=3)

        ttk.Label(index_frame, text="Keep Contours:").pack(side='left')

        # Min index
        ttk.Label(index_frame, text="From:").pack(side='left', padx=(10, 2))
        min_idx_spin = ttk.Spinbox(
            index_frame,
            from_=1,
            to=1000,
            width=5,
            textvariable=self.min_index
        )
        min_idx_spin.pack(side='left')

        # Max index
        ttk.Label(index_frame, text="To:").pack(side='left', padx=(10, 2))
        max_idx_spin = ttk.Spinbox(
            index_frame,
            from_=1,
            to=1000,
            width=5,
            textvariable=self.max_index
        )
        max_idx_spin.pack(side='left')

        ttk.Label(
            index_frame,
            text="(after sort)",
            font=('TkDefaultFont', 10, 'italic')
        ).pack(side='left', padx=5)

        # Contour count display
        count_frame = ttk.Frame(right_column)
        count_frame.pack(fill='x', pady=3)

        ttk.Label(count_frame, text="Contours found:").pack(side='left')
        self.count_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.count_label.pack(side='left', padx=5)

        ttk.Label(count_frame, text="Displayed:").pack(side='left', padx=(15, 0))
        self.displayed_label = ttk.Label(count_frame, text="0", font=('TkDefaultFont', 10, 'bold'))
        self.displayed_label.pack(side='left', padx=5)

        # Drawing mode dropdown
        draw_frame = ttk.Frame(right_column)
        draw_frame.pack(fill='x', pady=3)

        ttk.Label(draw_frame, text="Draw Mode:").pack(side='left')

        draw_values = [name for name, _ in self.DRAW_MODES]
        self.draw_combo = ttk.Combobox(
            draw_frame,
            values=draw_values,
            state='readonly',
            width=25
        )
        self.draw_combo.current(0)
        self.draw_combo.pack(side='left', padx=5)
        self.draw_combo.bind('<<ComboboxSelected>>', self._on_draw_mode_change)

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

    def _on_draw_mode_change(self, event):
        """Handle draw mode change"""
        self.draw_mode.set(self.draw_combo.current())

    def _on_sort_change(self, event):
        """Handle sort method change"""
        self.sort_method.set(self.sort_combo.current())

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []

        # Background
        lines.append(f"Background: {'Original Image' if self.show_original.get() else 'Black'}")

        # Threshold
        lines.append(f"Threshold: {self.threshold_value.get()}")

        # Retrieval mode
        retrieval_idx = self.retrieval_mode.get()
        retrieval_name = self.RETRIEVAL_MODES[retrieval_idx][1] if retrieval_idx < len(self.RETRIEVAL_MODES) else "Unknown"
        lines.append(f"Retrieval: {retrieval_name}")

        # Approximation method
        approx_idx = self.approx_method.get()
        approx_name = self.APPROX_METHODS[approx_idx][1] if approx_idx < len(self.APPROX_METHODS) else "Unknown"
        lines.append(f"Approximation: {approx_name}")

        # Sort method
        sort_idx = self.sort_method.get()
        sort_name = self.SORT_METHODS[sort_idx][0] if sort_idx < len(self.SORT_METHODS) else "Unknown"
        lines.append(f"Sort By: {sort_name}")

        # Draw mode
        draw_idx = self.draw_mode.get()
        draw_name = self.DRAW_MODES[draw_idx][0] if draw_idx < len(self.DRAW_MODES) else "Unknown"
        lines.append(f"Draw Mode: {draw_name}")

        return '\n'.join(lines)

    def _get_contour_centroid(self, contour):
        """Get centroid of a contour"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            # Fallback to bounding rect center
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w/2, y + h/2
        return cx, cy

    def _get_contour_circularity(self, contour):
        """Get circularity: 4*pi*area / perimeter^2 (1 = perfect circle)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter * perimeter)

    def _get_contour_aspect_ratio(self, contour):
        """Get aspect ratio: width / height of bounding rect"""
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return 0
        return w / h

    def _get_contour_extent(self, contour):
        """Get extent: contour area / bounding rect area"""
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area == 0:
            return 0
        return area / rect_area

    def _get_contour_solidity(self, contour):
        """Get solidity: contour area / convex hull area"""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return 0
        return area / hull_area

    def _sort_contours(self, contours, method_key):
        """Sort contours by the specified method"""
        if method_key is None or len(contours) == 0:
            return contours

        if method_key == "tb_lr":
            # Top-Bottom-Left-Right (Y primary, X secondary)
            return sorted(contours, key=lambda c: (self._get_contour_centroid(c)[1], self._get_contour_centroid(c)[0]))

        elif method_key == "lr_tb":
            # Left-Right-Top-Bottom (X primary, Y secondary)
            return sorted(contours, key=lambda c: (self._get_contour_centroid(c)[0], self._get_contour_centroid(c)[1]))

        elif method_key == "area_desc":
            return sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

        elif method_key == "area_asc":
            return sorted(contours, key=lambda c: cv2.contourArea(c))

        elif method_key == "perim_desc":
            return sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

        elif method_key == "perim_asc":
            return sorted(contours, key=lambda c: cv2.arcLength(c, True))

        elif method_key == "circ_desc":
            return sorted(contours, key=lambda c: self._get_contour_circularity(c), reverse=True)

        elif method_key == "circ_asc":
            return sorted(contours, key=lambda c: self._get_contour_circularity(c))

        elif method_key == "aspect_desc":
            return sorted(contours, key=lambda c: self._get_contour_aspect_ratio(c), reverse=True)

        elif method_key == "aspect_asc":
            return sorted(contours, key=lambda c: self._get_contour_aspect_ratio(c))

        elif method_key == "extent_desc":
            return sorted(contours, key=lambda c: self._get_contour_extent(c), reverse=True)

        elif method_key == "extent_asc":
            return sorted(contours, key=lambda c: self._get_contour_extent(c))

        elif method_key == "solid_desc":
            return sorted(contours, key=lambda c: self._get_contour_solidity(c), reverse=True)

        elif method_key == "solid_asc":
            return sorted(contours, key=lambda c: self._get_contour_solidity(c))

        return contours

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

        # Update count display (before sorting/index selection)
        if hasattr(self, 'count_label'):
            self.count_label.config(text=str(len(contours)))

        # Sort contours
        sort_idx = self.sort_method.get()
        sort_key = self.SORT_METHODS[sort_idx][1]
        sorted_contours = self._sort_contours(list(contours), sort_key)

        # Select contours by index (1-based for user, convert to 0-based)
        try:
            min_idx = max(0, int(self.min_index.get()) - 1)  # Convert to 0-based
            max_idx = int(self.max_index.get())  # Keep as-is since slicing is exclusive
        except (ValueError, tk.TclError):
            min_idx = 0
            max_idx = 100

        selected_contours = list(sorted_contours[min_idx:max_idx])

        # Update displayed count
        if hasattr(self, 'displayed_label'):
            self.displayed_label.config(text=str(len(selected_contours)))

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

        # Get draw mode
        try:
            draw_mode_idx = self.draw_mode.get()
            draw_mode_key = self.DRAW_MODES[draw_mode_idx][1]
        except:
            draw_mode_key = "contours"

        # Draw based on mode
        if draw_mode_key == "contours" or draw_mode_key is None:
            if len(selected_contours) > 0:
                cv2.drawContours(result, selected_contours, -1, color, thickness)

        elif draw_mode_key == "bounding_rect":
            for contour in selected_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        elif draw_mode_key == "rotated_rect":
            for contour in selected_contours:
                if len(contour) >= 5:  # minAreaRect needs at least 5 points
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(result, [box], 0, color, thickness)
                else:
                    # Fall back to regular bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        elif draw_mode_key == "enclosing_circle":
            for contour in selected_contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result, center, radius, color, thickness)

        elif draw_mode_key == "fitted_ellipse":
            for contour in selected_contours:
                if len(contour) >= 5:  # fitEllipse needs at least 5 points
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(result, ellipse, color, thickness)
                else:
                    # Fall back to enclosing circle
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(result, center, radius, color, thickness)

        elif draw_mode_key == "convex_hull":
            for contour in selected_contours:
                hull = cv2.convexHull(contour)
                cv2.drawContours(result, [hull], 0, color, thickness)

        elif draw_mode_key == "centroids":
            for contour in selected_contours:
                cx, cy = self._get_contour_centroid(contour)
                center = (int(cx), int(cy))
                # Draw a filled circle for centroid
                cv2.circle(result, center, thickness + 2, color, -1)

        return result

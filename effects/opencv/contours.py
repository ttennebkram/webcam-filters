"""
Contour detection and drawing effect using OpenCV.

Finds contours in images and draws them with various options.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


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

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Show Original Background', 'key': 'show_original', 'default': True},
            {'type': 'slider', 'label': 'Threshold', 'key': 'threshold_value', 'min': 0, 'max': 255, 'default': 127},
            {'type': 'dropdown', 'label': 'Retrieval Mode', 'key': 'retrieval_mode', 'options': [name for _, name, _ in self.RETRIEVAL_MODES], 'default': 'RETR_EXTERNAL'},
            {'type': 'dropdown', 'label': 'Approximation', 'key': 'approx_method', 'options': [name for _, name, _ in self.APPROX_METHODS], 'default': 'CHAIN_APPROX_NONE'},
            {'type': 'slider', 'label': 'Line Thickness', 'key': 'thickness', 'min': 1, 'max': 10, 'default': 2},
            {'type': 'dropdown', 'label': 'Sort By', 'key': 'sort_method', 'options': [name for name, _ in self.SORT_METHODS], 'default': 'None'},
            {'type': 'dropdown', 'label': 'Draw Mode', 'key': 'draw_mode', 'options': [name for name, _ in self.DRAW_MODES], 'default': 'Contours'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        retrieval_idx = self.retrieval_mode.get()
        approx_idx = self.approx_method.get()
        sort_idx = self.sort_method.get()
        draw_idx = self.draw_mode.get()

        return {
            'show_original': self.show_original.get(),
            'threshold_value': self.threshold_value.get(),
            'retrieval_mode': self.RETRIEVAL_MODES[retrieval_idx][1] if retrieval_idx < len(self.RETRIEVAL_MODES) else 'RETR_EXTERNAL',
            'approx_method': self.APPROX_METHODS[approx_idx][1] if approx_idx < len(self.APPROX_METHODS) else 'CHAIN_APPROX_NONE',
            'thickness': self.thickness.get(),
            'sort_method': self.SORT_METHODS[sort_idx][0] if sort_idx < len(self.SORT_METHODS) else 'None',
            'draw_mode': self.DRAW_MODES[draw_idx][0] if draw_idx < len(self.DRAW_MODES) else 'Contours',
        }

    def create_control_panel(self, parent, mode='view'):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)
        self._control_parent = parent
        self._current_mode = mode

        # Create the EffectForm
        schema = self.get_form_schema()
        self._subform = Subform(schema)

        self._effect_form = EffectForm(
            effect_name=self.get_name(),
            subform=self._subform,
            enabled_var=self.enabled,
            description=self.get_description(),
            signature=self.get_method_signature(),
            on_mode_toggle=self._toggle_mode,
            on_copy_text=self._copy_text,
            on_copy_json=self._copy_json,
            on_paste_text=self._paste_text,
            on_paste_json=self._paste_json,
            on_add_below=getattr(self, '_on_add_below', None),
            on_remove=getattr(self, '_on_remove', None)
        )

        # Render the form
        form_frame = self._effect_form.render(
            self.control_panel,
            mode=mode,
            data=self.get_current_data()
        )
        form_frame.pack(fill='both', expand=True)

        # Store reference to subform for syncing values back
        self._update_vars_from_subform()

        return self.control_panel

    def _update_vars_from_subform(self):
        """Set up tracing to sync subform values back to effect variables"""
        # When subform values change, update effect's tk.Variables
        for key, var in self._subform._vars.items():
            if key == 'show_original':
                var.trace_add('write', lambda *args: self.show_original.set(self._subform._vars['show_original'].get()))
            elif key == 'threshold_value':
                var.trace_add('write', lambda *args: self.threshold_value.set(int(self._subform._vars['threshold_value'].get())))
            elif key == 'retrieval_mode':
                var.trace_add('write', lambda *args: self._update_retrieval_mode())
            elif key == 'approx_method':
                var.trace_add('write', lambda *args: self._update_approx_method())
            elif key == 'thickness':
                var.trace_add('write', lambda *args: self.thickness.set(int(self._subform._vars['thickness'].get())))
            elif key == 'sort_method':
                var.trace_add('write', lambda *args: self._update_sort_method())
            elif key == 'draw_mode':
                var.trace_add('write', lambda *args: self._update_draw_mode())

    def _update_retrieval_mode(self):
        """Update retrieval mode from subform value"""
        value = self._subform._vars['retrieval_mode'].get()
        for idx, (_, name, _) in enumerate(self.RETRIEVAL_MODES):
            if name == value:
                self.retrieval_mode.set(idx)
                break

    def _update_approx_method(self):
        """Update approximation method from subform value"""
        value = self._subform._vars['approx_method'].get()
        for idx, (_, name, _) in enumerate(self.APPROX_METHODS):
            if name == value:
                self.approx_method.set(idx)
                break

    def _update_sort_method(self):
        """Update sort method from subform value"""
        value = self._subform._vars['sort_method'].get()
        for idx, (name, _) in enumerate(self.SORT_METHODS):
            if name == value:
                self.sort_method.set(idx)
                break

    def _update_draw_mode(self):
        """Update draw mode from subform value"""
        value = self._subform._vars['draw_mode'].get()
        for idx, (name, _) in enumerate(self.DRAW_MODES):
            if name == value:
                self.draw_mode.set(idx)
                break

    def _toggle_mode(self):
        """Toggle between edit and view modes"""
        self._current_mode = 'view' if self._current_mode == 'edit' else 'edit'

        # Notify pipeline when switching to edit mode
        if self._current_mode == 'edit' and hasattr(self, '_on_edit') and self._on_edit:
            self._on_edit()

        # Re-render the entire control panel
        for child in self.control_panel.winfo_children():
            child.destroy()

        schema = self.get_form_schema()
        self._subform = Subform(schema)

        self._effect_form = EffectForm(
            effect_name=self.get_name(),
            subform=self._subform,
            enabled_var=self.enabled,
            description=self.get_description(),
            signature=self.get_method_signature(),
            on_mode_toggle=self._toggle_mode,
            on_copy_text=self._copy_text,
            on_copy_json=self._copy_json,
            on_paste_text=self._paste_text,
            on_paste_json=self._paste_json,
            on_add_below=getattr(self, '_on_add_below', None),
            on_remove=getattr(self, '_on_remove', None)
        )

        form_frame = self._effect_form.render(
            self.control_panel,
            mode=self._current_mode,
            data=self.get_current_data()
        )
        form_frame.pack(fill='both', expand=True)

        if self._current_mode == 'edit':
            self._update_vars_from_subform()

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())
        lines.append(f"Background: {'Original Image' if self.show_original.get() else 'Black'}")
        lines.append(f"Threshold: {self.threshold_value.get()}")

        retrieval_idx = self.retrieval_mode.get()
        lines.append(f"Retrieval: {self.RETRIEVAL_MODES[retrieval_idx][1] if retrieval_idx < len(self.RETRIEVAL_MODES) else 'Unknown'}")

        approx_idx = self.approx_method.get()
        lines.append(f"Approximation: {self.APPROX_METHODS[approx_idx][1] if approx_idx < len(self.APPROX_METHODS) else 'Unknown'}")

        lines.append(f"Thickness: {self.thickness.get()}")

        sort_idx = self.sort_method.get()
        lines.append(f"Sort By: {self.SORT_METHODS[sort_idx][0] if sort_idx < len(self.SORT_METHODS) else 'Unknown'}")

        draw_idx = self.draw_mode.get()
        lines.append(f"Draw Mode: {self.DRAW_MODES[draw_idx][0] if draw_idx < len(self.DRAW_MODES) else 'Unknown'}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        retrieval_idx = self.retrieval_mode.get()
        approx_idx = self.approx_method.get()
        sort_idx = self.sort_method.get()
        draw_idx = self.draw_mode.get()

        data = {
            'effect': self.get_name(),
            'show_original': self.show_original.get(),
            'threshold_value': self.threshold_value.get(),
            'retrieval_mode': self.RETRIEVAL_MODES[retrieval_idx][1] if retrieval_idx < len(self.RETRIEVAL_MODES) else 'RETR_EXTERNAL',
            'approx_method': self.APPROX_METHODS[approx_idx][1] if approx_idx < len(self.APPROX_METHODS) else 'CHAIN_APPROX_NONE',
            'thickness': self.thickness.get(),
            'sort_method': self.SORT_METHODS[sort_idx][0] if sort_idx < len(self.SORT_METHODS) else 'None',
            'draw_mode': self.DRAW_MODES[draw_idx][0] if draw_idx < len(self.DRAW_MODES) else 'Contours',
        }

        text = json.dumps(data, indent=2)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _paste_text(self):
        """Paste settings from human-readable text on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            lines = text.strip().split('\n')

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if 'background' in key:
                        self.show_original.set(value.lower() != 'black')
                    elif 'threshold' in key:
                        self.threshold_value.set(max(0, min(255, int(value))))
                    elif 'retrieval' in key:
                        for idx, (_, name, _) in enumerate(self.RETRIEVAL_MODES):
                            if name == value:
                                self.retrieval_mode.set(idx)
                                break
                    elif 'approximation' in key:
                        for idx, (_, name, _) in enumerate(self.APPROX_METHODS):
                            if name == value:
                                self.approx_method.set(idx)
                                break
                    elif 'thickness' in key:
                        self.thickness.set(max(1, min(10, int(value))))
                    elif 'sort' in key:
                        for idx, (name, _) in enumerate(self.SORT_METHODS):
                            if name == value:
                                self.sort_method.set(idx)
                                break
                    elif 'draw' in key and 'mode' in key:
                        for idx, (name, _) in enumerate(self.DRAW_MODES):
                            if name == value:
                                self.draw_mode.set(idx)
                                break

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                self._sync_subform_from_vars()
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'show_original' in data:
                self.show_original.set(bool(data['show_original']))
            if 'threshold_value' in data:
                self.threshold_value.set(max(0, min(255, int(data['threshold_value']))))
            if 'retrieval_mode' in data:
                for idx, (_, name, _) in enumerate(self.RETRIEVAL_MODES):
                    if name == data['retrieval_mode']:
                        self.retrieval_mode.set(idx)
                        break
            if 'approx_method' in data:
                for idx, (_, name, _) in enumerate(self.APPROX_METHODS):
                    if name == data['approx_method']:
                        self.approx_method.set(idx)
                        break
            if 'thickness' in data:
                self.thickness.set(max(1, min(10, int(data['thickness']))))
            if 'sort_method' in data:
                for idx, (name, _) in enumerate(self.SORT_METHODS):
                    if name == data['sort_method']:
                        self.sort_method.set(idx)
                        break
            if 'draw_mode' in data:
                for idx, (name, _) in enumerate(self.DRAW_MODES):
                    if name == data['draw_mode']:
                        self.draw_mode.set(idx)
                        break

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                self._sync_subform_from_vars()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_vars(self):
        """Sync subform variables from effect variables"""
        if 'show_original' in self._subform._vars:
            self._subform._vars['show_original'].set(self.show_original.get())
        if 'threshold_value' in self._subform._vars:
            self._subform._vars['threshold_value'].set(self.threshold_value.get())
        if 'retrieval_mode' in self._subform._vars:
            idx = self.retrieval_mode.get()
            self._subform._vars['retrieval_mode'].set(self.RETRIEVAL_MODES[idx][1] if idx < len(self.RETRIEVAL_MODES) else 'RETR_EXTERNAL')
        if 'approx_method' in self._subform._vars:
            idx = self.approx_method.get()
            self._subform._vars['approx_method'].set(self.APPROX_METHODS[idx][1] if idx < len(self.APPROX_METHODS) else 'CHAIN_APPROX_NONE')
        if 'thickness' in self._subform._vars:
            self._subform._vars['thickness'].set(self.thickness.get())
        if 'sort_method' in self._subform._vars:
            idx = self.sort_method.get()
            self._subform._vars['sort_method'].set(self.SORT_METHODS[idx][0] if idx < len(self.SORT_METHODS) else 'None')
        if 'draw_mode' in self._subform._vars:
            idx = self.draw_mode.get()
            self._subform._vars['draw_mode'].set(self.DRAW_MODES[idx][0] if idx < len(self.DRAW_MODES) else 'Contours')

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

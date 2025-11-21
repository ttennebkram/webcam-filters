"""
Hough Circle detection effect using OpenCV.

Detects circles in images using the Hough Transform.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class HoughCirclesEffect(BaseUIEffect):
    """Detect and draw circles using Hough Transform"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_circles = None

        # HoughCircles parameters
        self.dp = tk.DoubleVar(value=1.0)  # Inverse ratio of accumulator resolution
        self.min_dist = tk.IntVar(value=50)  # Min distance between circle centers
        self.param1 = tk.IntVar(value=100)  # Canny high threshold
        self.param2 = tk.IntVar(value=30)  # Accumulator threshold
        self.min_radius = tk.IntVar(value=10)
        self.max_radius = tk.IntVar(value=100)

        # Drawing options
        self.circle_color_b = tk.IntVar(value=0)
        self.circle_color_g = tk.IntVar(value=255)
        self.circle_color_r = tk.IntVar(value=0)
        self.thickness = tk.IntVar(value=2)
        self.draw_center = tk.BooleanVar(value=True)

    @classmethod
    def get_name(cls) -> str:
        return "Hough Circles"

    @classmethod
    def get_description(cls) -> str:
        return "Detect circles using Hough Transform"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.HoughCircles(image, method, dp, minDist)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Show Original', 'key': 'show_original', 'default': True},
            {'type': 'checkbox', 'label': 'Draw Features', 'key': 'draw_features', 'default': True},
            {'type': 'slider', 'label': 'Min Distance', 'key': 'min_dist', 'min': 1, 'max': 200, 'default': 50},
            {'type': 'slider', 'label': 'Param1 (Canny)', 'key': 'param1', 'min': 1, 'max': 300, 'default': 100},
            {'type': 'slider', 'label': 'Param2 (Accum)', 'key': 'param2', 'min': 1, 'max': 100, 'default': 30},
            {'type': 'slider', 'label': 'Min Radius', 'key': 'min_radius', 'min': 0, 'max': 200, 'default': 10},
            {'type': 'slider', 'label': 'Max Radius', 'key': 'max_radius', 'min': 0, 'max': 500, 'default': 100},
            {'type': 'slider', 'label': 'Thickness', 'key': 'thickness', 'min': 1, 'max': 10, 'default': 2},
            {'type': 'checkbox', 'label': 'Draw Center', 'key': 'draw_center', 'default': True},
            {'type': 'slider', 'label': 'Color B', 'key': 'circle_color_b', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color G', 'key': 'circle_color_g', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'slider', 'label': 'Color R', 'key': 'circle_color_r', 'min': 0, 'max': 255, 'default': 0},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'min_dist': self.min_dist.get(),
            'param1': self.param1.get(),
            'param2': self.param2.get(),
            'min_radius': self.min_radius.get(),
            'max_radius': self.max_radius.get(),
            'thickness': self.thickness.get(),
            'draw_center': self.draw_center.get(),
            'circle_color_b': self.circle_color_b.get(),
            'circle_color_g': self.circle_color_g.get(),
            'circle_color_r': self.circle_color_r.get()
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
            elif key == 'draw_features':
                var.trace_add('write', lambda *args: self.draw_features.set(self._subform._vars['draw_features'].get()))
            elif key == 'min_dist':
                var.trace_add('write', lambda *args: self.min_dist.set(int(self._subform._vars['min_dist'].get())))
            elif key == 'param1':
                var.trace_add('write', lambda *args: self.param1.set(int(self._subform._vars['param1'].get())))
            elif key == 'param2':
                var.trace_add('write', lambda *args: self.param2.set(int(self._subform._vars['param2'].get())))
            elif key == 'min_radius':
                var.trace_add('write', lambda *args: self.min_radius.set(int(self._subform._vars['min_radius'].get())))
            elif key == 'max_radius':
                var.trace_add('write', lambda *args: self.max_radius.set(int(self._subform._vars['max_radius'].get())))
            elif key == 'thickness':
                var.trace_add('write', lambda *args: self.thickness.set(int(self._subform._vars['thickness'].get())))
            elif key == 'draw_center':
                var.trace_add('write', lambda *args: self.draw_center.set(self._subform._vars['draw_center'].get()))
            elif key == 'circle_color_b':
                var.trace_add('write', lambda *args: self.circle_color_b.set(int(self._subform._vars['circle_color_b'].get())))
            elif key == 'circle_color_g':
                var.trace_add('write', lambda *args: self.circle_color_g.set(int(self._subform._vars['circle_color_g'].get())))
            elif key == 'circle_color_r':
                var.trace_add('write', lambda *args: self.circle_color_r.set(int(self._subform._vars['circle_color_r'].get())))

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
        lines.append(f"Show Original: {'Yes' if self.show_original.get() else 'No'}")
        lines.append(f"Draw Features: {'Yes' if self.draw_features.get() else 'No'}")
        lines.append(f"Min Distance: {self.min_dist.get()}")
        lines.append(f"Param1 (Canny): {self.param1.get()}")
        lines.append(f"Param2 (Accum): {self.param2.get()}")
        lines.append(f"Min Radius: {self.min_radius.get()}")
        lines.append(f"Max Radius: {self.max_radius.get()}")
        lines.append(f"Thickness: {self.thickness.get()}")
        lines.append(f"Draw Center: {'Yes' if self.draw_center.get() else 'No'}")
        lines.append(f"Color BGR: ({self.circle_color_b.get()}, {self.circle_color_g.get()}, {self.circle_color_r.get()})")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'min_dist': self.min_dist.get(),
            'param1': self.param1.get(),
            'param2': self.param2.get(),
            'min_radius': self.min_radius.get(),
            'max_radius': self.max_radius.get(),
            'thickness': self.thickness.get(),
            'draw_center': self.draw_center.get(),
            'circle_color_b': self.circle_color_b.get(),
            'circle_color_g': self.circle_color_g.get(),
            'circle_color_r': self.circle_color_r.get()
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

                    if 'show original' in key:
                        self.show_original.set(value.lower() in ('yes', 'true', '1'))
                    elif 'draw features' in key:
                        self.draw_features.set(value.lower() in ('yes', 'true', '1'))
                    elif 'min distance' in key:
                        self.min_dist.set(max(1, min(200, int(value))))
                    elif 'param1' in key or 'canny' in key:
                        self.param1.set(max(1, min(300, int(value))))
                    elif 'param2' in key or 'accum' in key:
                        self.param2.set(max(1, min(100, int(value))))
                    elif 'min radius' in key:
                        self.min_radius.set(max(0, min(200, int(value))))
                    elif 'max radius' in key:
                        self.max_radius.set(max(0, min(500, int(value))))
                    elif 'thickness' in key:
                        self.thickness.set(max(1, min(10, int(value))))
                    elif 'draw center' in key:
                        self.draw_center.set(value.lower() in ('yes', 'true', '1'))
                    elif 'color bgr' in key:
                        # Parse (b, g, r) format
                        value = value.strip('()')
                        parts = [int(x.strip()) for x in value.split(',')]
                        if len(parts) >= 3:
                            self.circle_color_b.set(max(0, min(255, parts[0])))
                            self.circle_color_g.set(max(0, min(255, parts[1])))
                            self.circle_color_r.set(max(0, min(255, parts[2])))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['show_original', 'draw_features', 'min_dist', 'param1', 'param2', 'min_radius', 'max_radius', 'thickness', 'draw_center', 'circle_color_b', 'circle_color_g', 'circle_color_r']:
                    if key in self._subform._vars:
                        if key in ['show_original', 'draw_features', 'draw_center']:
                            self._subform._vars[key].set(getattr(self, key).get())
                        else:
                            self._subform._vars[key].set(getattr(self, key).get())
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
            if 'draw_features' in data:
                self.draw_features.set(bool(data['draw_features']))
            if 'min_dist' in data:
                self.min_dist.set(max(1, min(200, int(data['min_dist']))))
            if 'param1' in data:
                self.param1.set(max(1, min(300, int(data['param1']))))
            if 'param2' in data:
                self.param2.set(max(1, min(100, int(data['param2']))))
            if 'min_radius' in data:
                self.min_radius.set(max(0, min(200, int(data['min_radius']))))
            if 'max_radius' in data:
                self.max_radius.set(max(0, min(500, int(data['max_radius']))))
            if 'thickness' in data:
                self.thickness.set(max(1, min(10, int(data['thickness']))))
            if 'draw_center' in data:
                self.draw_center.set(bool(data['draw_center']))
            if 'circle_color_b' in data:
                self.circle_color_b.set(max(0, min(255, int(data['circle_color_b']))))
            if 'circle_color_g' in data:
                self.circle_color_g.set(max(0, min(255, int(data['circle_color_g']))))
            if 'circle_color_r' in data:
                self.circle_color_r.set(max(0, min(255, int(data['circle_color_r']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['show_original', 'draw_features', 'min_dist', 'param1', 'param2', 'min_radius', 'max_radius', 'thickness', 'draw_center', 'circle_color_b', 'circle_color_g', 'circle_color_r']:
                    if key in self._subform._vars:
                        if key in ['show_original', 'draw_features', 'draw_center']:
                            self._subform._vars[key].set(getattr(self, key).get())
                        else:
                            self._subform._vars[key].set(getattr(self, key).get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Input: {'Original Image' if self.show_original.get() else 'Black'}")
        lines.append(f"Output: {'Draw Features' if self.draw_features.get() else 'Raw Values Only'}")
        lines.append(f"Min Distance: {self.min_dist.get()}")
        lines.append(f"Param1: {self.param1.get()}")
        lines.append(f"Param2: {self.param2.get()}")
        lines.append(f"Radius: {self.min_radius.get()} - {self.max_radius.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw circles on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        dp = self.dp.get()
        min_dist = self.min_dist.get()
        param1 = self.param1.get()
        param2 = self.param2.get()
        min_radius = self.min_radius.get()
        max_radius = self.max_radius.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.circle_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.circle_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.circle_color_r.get()
        except Exception:
            color_r = 0
        color = (color_b, color_g, color_r)

        thickness = self.thickness.get()

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp,
            min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        # Store raw values for pipeline use
        self.detected_circles = circles

        circles_count = 0

        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles_count = len(circles[0])

            if self.draw_features.get():
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]

                    # Draw circle
                    cv2.circle(result, center, radius, color, thickness)

                    # Draw center point
                    if self.draw_center.get():
                        cv2.circle(result, center, 2, color, 3)

        return result

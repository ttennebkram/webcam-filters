"""
Hough Line detection effect using OpenCV.

Detects straight lines in images using the Hough Transform.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


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

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Probabilistic', 'key': 'use_probabilistic', 'default': True},
            {'type': 'checkbox', 'label': 'Show Original', 'key': 'show_original', 'default': True},
            {'type': 'checkbox', 'label': 'Draw Features', 'key': 'draw_features', 'default': True},
            {'type': 'slider', 'label': 'Canny Thresh 1', 'key': 'canny_thresh1', 'min': 0, 'max': 255, 'default': 50},
            {'type': 'slider', 'label': 'Canny Thresh 2', 'key': 'canny_thresh2', 'min': 0, 'max': 255, 'default': 150},
            {'type': 'slider', 'label': 'Threshold', 'key': 'threshold', 'min': 1, 'max': 200, 'default': 50},
            {'type': 'slider', 'label': 'Min Line Length', 'key': 'min_line_length', 'min': 1, 'max': 200, 'default': 50},
            {'type': 'slider', 'label': 'Max Line Gap', 'key': 'max_line_gap', 'min': 1, 'max': 100, 'default': 10},
            {'type': 'slider', 'label': 'Max Lines', 'key': 'max_lines', 'min': 1, 'max': 500, 'default': 100},
            {'type': 'slider', 'label': 'Thickness', 'key': 'thickness', 'min': 1, 'max': 10, 'default': 2},
            {'type': 'slider', 'label': 'Color B', 'key': 'line_color_b', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color G', 'key': 'line_color_g', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color R', 'key': 'line_color_r', 'min': 0, 'max': 255, 'default': 255},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'use_probabilistic': self.use_probabilistic.get(),
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'canny_thresh1': self.canny_thresh1.get(),
            'canny_thresh2': self.canny_thresh2.get(),
            'threshold': self.threshold.get(),
            'min_line_length': self.min_line_length.get(),
            'max_line_gap': self.max_line_gap.get(),
            'max_lines': self.max_lines.get(),
            'thickness': self.thickness.get(),
            'line_color_b': self.line_color_b.get(),
            'line_color_g': self.line_color_g.get(),
            'line_color_r': self.line_color_r.get()
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
            if key == 'use_probabilistic':
                var.trace_add('write', lambda *args: self.use_probabilistic.set(self._subform._vars['use_probabilistic'].get()))
            elif key == 'show_original':
                var.trace_add('write', lambda *args: self.show_original.set(self._subform._vars['show_original'].get()))
            elif key == 'draw_features':
                var.trace_add('write', lambda *args: self.draw_features.set(self._subform._vars['draw_features'].get()))
            elif key == 'canny_thresh1':
                var.trace_add('write', lambda *args: self.canny_thresh1.set(int(self._subform._vars['canny_thresh1'].get())))
            elif key == 'canny_thresh2':
                var.trace_add('write', lambda *args: self.canny_thresh2.set(int(self._subform._vars['canny_thresh2'].get())))
            elif key == 'threshold':
                var.trace_add('write', lambda *args: self.threshold.set(int(self._subform._vars['threshold'].get())))
            elif key == 'min_line_length':
                var.trace_add('write', lambda *args: self.min_line_length.set(int(self._subform._vars['min_line_length'].get())))
            elif key == 'max_line_gap':
                var.trace_add('write', lambda *args: self.max_line_gap.set(int(self._subform._vars['max_line_gap'].get())))
            elif key == 'max_lines':
                var.trace_add('write', lambda *args: self.max_lines.set(int(self._subform._vars['max_lines'].get())))
            elif key == 'thickness':
                var.trace_add('write', lambda *args: self.thickness.set(int(self._subform._vars['thickness'].get())))
            elif key == 'line_color_b':
                var.trace_add('write', lambda *args: self.line_color_b.set(int(self._subform._vars['line_color_b'].get())))
            elif key == 'line_color_g':
                var.trace_add('write', lambda *args: self.line_color_g.set(int(self._subform._vars['line_color_g'].get())))
            elif key == 'line_color_r':
                var.trace_add('write', lambda *args: self.line_color_r.set(int(self._subform._vars['line_color_r'].get())))

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
        lines.append(f"Method: {'Probabilistic' if self.use_probabilistic.get() else 'Standard'}")
        lines.append(f"Show Original: {'Yes' if self.show_original.get() else 'No'}")
        lines.append(f"Draw Features: {'Yes' if self.draw_features.get() else 'No'}")
        lines.append(f"Canny Thresh 1: {self.canny_thresh1.get()}")
        lines.append(f"Canny Thresh 2: {self.canny_thresh2.get()}")
        lines.append(f"Threshold: {self.threshold.get()}")
        lines.append(f"Min Line Length: {self.min_line_length.get()}")
        lines.append(f"Max Line Gap: {self.max_line_gap.get()}")
        lines.append(f"Max Lines: {self.max_lines.get()}")
        lines.append(f"Thickness: {self.thickness.get()}")
        lines.append(f"Color BGR: ({self.line_color_b.get()}, {self.line_color_g.get()}, {self.line_color_r.get()})")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'use_probabilistic': self.use_probabilistic.get(),
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'canny_thresh1': self.canny_thresh1.get(),
            'canny_thresh2': self.canny_thresh2.get(),
            'threshold': self.threshold.get(),
            'min_line_length': self.min_line_length.get(),
            'max_line_gap': self.max_line_gap.get(),
            'max_lines': self.max_lines.get(),
            'thickness': self.thickness.get(),
            'line_color_b': self.line_color_b.get(),
            'line_color_g': self.line_color_g.get(),
            'line_color_r': self.line_color_r.get()
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

                    if 'method' in key:
                        self.use_probabilistic.set(value.lower() == 'probabilistic')
                    elif 'show original' in key:
                        self.show_original.set(value.lower() in ('yes', 'true', '1'))
                    elif 'draw features' in key:
                        self.draw_features.set(value.lower() in ('yes', 'true', '1'))
                    elif 'canny thresh 1' in key:
                        self.canny_thresh1.set(max(0, min(255, int(value))))
                    elif 'canny thresh 2' in key:
                        self.canny_thresh2.set(max(0, min(255, int(value))))
                    elif key == 'threshold':
                        self.threshold.set(max(1, min(200, int(value))))
                    elif 'min line length' in key:
                        self.min_line_length.set(max(1, min(200, int(value))))
                    elif 'max line gap' in key:
                        self.max_line_gap.set(max(1, min(100, int(value))))
                    elif 'max lines' in key:
                        self.max_lines.set(max(1, min(500, int(value))))
                    elif 'thickness' in key:
                        self.thickness.set(max(1, min(10, int(value))))
                    elif 'color bgr' in key:
                        # Parse (b, g, r) format
                        value = value.strip('()')
                        parts = [int(x.strip()) for x in value.split(',')]
                        if len(parts) >= 3:
                            self.line_color_b.set(max(0, min(255, parts[0])))
                            self.line_color_g.set(max(0, min(255, parts[1])))
                            self.line_color_r.set(max(0, min(255, parts[2])))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['use_probabilistic', 'show_original', 'draw_features', 'canny_thresh1', 'canny_thresh2', 'threshold', 'min_line_length', 'max_line_gap', 'max_lines', 'thickness', 'line_color_b', 'line_color_g', 'line_color_r']:
                    if key in self._subform._vars:
                        if key in ['use_probabilistic', 'show_original', 'draw_features']:
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

            if 'use_probabilistic' in data:
                self.use_probabilistic.set(bool(data['use_probabilistic']))
            if 'show_original' in data:
                self.show_original.set(bool(data['show_original']))
            if 'draw_features' in data:
                self.draw_features.set(bool(data['draw_features']))
            if 'canny_thresh1' in data:
                self.canny_thresh1.set(max(0, min(255, int(data['canny_thresh1']))))
            if 'canny_thresh2' in data:
                self.canny_thresh2.set(max(0, min(255, int(data['canny_thresh2']))))
            if 'threshold' in data:
                self.threshold.set(max(1, min(200, int(data['threshold']))))
            if 'min_line_length' in data:
                self.min_line_length.set(max(1, min(200, int(data['min_line_length']))))
            if 'max_line_gap' in data:
                self.max_line_gap.set(max(1, min(100, int(data['max_line_gap']))))
            if 'max_lines' in data:
                self.max_lines.set(max(1, min(500, int(data['max_lines']))))
            if 'thickness' in data:
                self.thickness.set(max(1, min(10, int(data['thickness']))))
            if 'line_color_b' in data:
                self.line_color_b.set(max(0, min(255, int(data['line_color_b']))))
            if 'line_color_g' in data:
                self.line_color_g.set(max(0, min(255, int(data['line_color_g']))))
            if 'line_color_r' in data:
                self.line_color_r.set(max(0, min(255, int(data['line_color_r']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['use_probabilistic', 'show_original', 'draw_features', 'canny_thresh1', 'canny_thresh2', 'threshold', 'min_line_length', 'max_line_gap', 'max_lines', 'thickness', 'line_color_b', 'line_color_g', 'line_color_r']:
                    if key in self._subform._vars:
                        if key in ['use_probabilistic', 'show_original', 'draw_features']:
                            self._subform._vars[key].set(getattr(self, key).get())
                        else:
                            self._subform._vars[key].set(getattr(self, key).get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Method: {'Probabilistic' if self.use_probabilistic.get() else 'Standard'}")
        lines.append(f"Input: {'Original Image' if self.show_original.get() else 'Black'}")
        lines.append(f"Output: {'Draw Features' if self.draw_features.get() else 'Raw Values Only'}")
        lines.append(f"Canny: {self.canny_thresh1.get()} / {self.canny_thresh2.get()}")
        lines.append(f"Threshold: {self.threshold.get()}")
        lines.append(f"Min Length: {self.min_line_length.get()}")
        return '\n'.join(lines)

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

        return result

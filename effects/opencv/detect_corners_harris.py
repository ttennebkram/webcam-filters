"""
Harris corner detection effect using OpenCV.

Detects corners using the Harris corner detection algorithm.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class HarrisCornersEffect(BaseUIEffect):
    """Detect and mark corners using Harris corner detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_corners = None

        # Harris parameters
        self.block_size = tk.IntVar(value=2)  # Neighborhood size
        self.ksize = tk.IntVar(value=3)  # Sobel aperture
        self.k = tk.DoubleVar(value=0.04)  # Harris free parameter

        # Threshold for corner detection
        self.threshold = tk.DoubleVar(value=0.01)  # Fraction of max response

        # Drawing options
        self.corner_color_b = tk.IntVar(value=0)
        self.corner_color_g = tk.IntVar(value=0)
        self.corner_color_r = tk.IntVar(value=255)
        self.marker_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Detect Corners Harris"

    @classmethod
    def get_description(cls) -> str:
        return "Detect corners using Harris corner detection"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.cornerHarris(src, blockSize, ksize, k)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Show Original', 'key': 'show_original', 'default': True},
            {'type': 'checkbox', 'label': 'Draw Features', 'key': 'draw_features', 'default': True},
            {'type': 'slider', 'label': 'Block Size', 'key': 'block_size', 'min': 2, 'max': 10, 'default': 2},
            {'type': 'dropdown', 'label': 'Sobel Aperture', 'key': 'ksize', 'options': [3, 5, 7], 'default': 3},
            {'type': 'slider', 'label': 'Harris k', 'key': 'k', 'min': 0.01, 'max': 0.10, 'default': 0.04, 'resolution': 0.01},
            {'type': 'slider', 'label': 'Threshold', 'key': 'threshold', 'min': 0.001, 'max': 0.1, 'default': 0.01, 'resolution': 0.001},
            {'type': 'slider', 'label': 'Marker Size', 'key': 'marker_size', 'min': 1, 'max': 15, 'default': 5},
            {'type': 'slider', 'label': 'Color B', 'key': 'corner_color_b', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color G', 'key': 'corner_color_g', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color R', 'key': 'corner_color_r', 'min': 0, 'max': 255, 'default': 255},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'block_size': self.block_size.get(),
            'ksize': self.ksize.get(),
            'k': self.k.get(),
            'threshold': self.threshold.get(),
            'marker_size': self.marker_size.get(),
            'corner_color_b': self.corner_color_b.get(),
            'corner_color_g': self.corner_color_g.get(),
            'corner_color_r': self.corner_color_r.get()
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
            elif key == 'block_size':
                var.trace_add('write', lambda *args: self.block_size.set(int(self._subform._vars['block_size'].get())))
            elif key == 'ksize':
                var.trace_add('write', lambda *args: self.ksize.set(int(self._subform._vars['ksize'].get())))
            elif key == 'k':
                var.trace_add('write', lambda *args: self.k.set(float(self._subform._vars['k'].get())))
            elif key == 'threshold':
                var.trace_add('write', lambda *args: self.threshold.set(float(self._subform._vars['threshold'].get())))
            elif key == 'marker_size':
                var.trace_add('write', lambda *args: self.marker_size.set(int(self._subform._vars['marker_size'].get())))
            elif key == 'corner_color_b':
                var.trace_add('write', lambda *args: self.corner_color_b.set(int(self._subform._vars['corner_color_b'].get())))
            elif key == 'corner_color_g':
                var.trace_add('write', lambda *args: self.corner_color_g.set(int(self._subform._vars['corner_color_g'].get())))
            elif key == 'corner_color_r':
                var.trace_add('write', lambda *args: self.corner_color_r.set(int(self._subform._vars['corner_color_r'].get())))

    def _toggle_mode(self):
        """Toggle between edit and view modes"""
        self._current_mode = 'view' if self._current_mode == 'edit' else 'edit'

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
        lines.append(f"Block Size: {self.block_size.get()}")
        lines.append(f"Sobel Aperture: {self.ksize.get()}")
        lines.append(f"Harris k: {self.k.get():.2f}")
        lines.append(f"Threshold: {self.threshold.get():.3f}")
        lines.append(f"Marker Size: {self.marker_size.get()}")
        lines.append(f"Color BGR: ({self.corner_color_b.get()}, {self.corner_color_g.get()}, {self.corner_color_r.get()})")

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
            'block_size': self.block_size.get(),
            'ksize': self.ksize.get(),
            'k': self.k.get(),
            'threshold': self.threshold.get(),
            'marker_size': self.marker_size.get(),
            'corner_color_b': self.corner_color_b.get(),
            'corner_color_g': self.corner_color_g.get(),
            'corner_color_r': self.corner_color_r.get()
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
                    elif 'block size' in key:
                        self.block_size.set(max(2, min(10, int(value))))
                    elif 'sobel' in key or 'aperture' in key:
                        val = int(value)
                        if val <= 3:
                            val = 3
                        elif val <= 5:
                            val = 5
                        else:
                            val = 7
                        self.ksize.set(val)
                    elif 'harris k' in key:
                        self.k.set(max(0.01, min(0.10, float(value))))
                    elif 'threshold' in key:
                        self.threshold.set(max(0.001, min(0.1, float(value))))
                    elif 'marker' in key:
                        self.marker_size.set(max(1, min(15, int(value))))
                    elif 'color bgr' in key:
                        # Parse (b, g, r) format
                        value = value.strip('()')
                        parts = [int(x.strip()) for x in value.split(',')]
                        if len(parts) >= 3:
                            self.corner_color_b.set(max(0, min(255, parts[0])))
                            self.corner_color_g.set(max(0, min(255, parts[1])))
                            self.corner_color_r.set(max(0, min(255, parts[2])))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['show_original', 'draw_features', 'block_size', 'ksize', 'k', 'threshold', 'marker_size', 'corner_color_b', 'corner_color_g', 'corner_color_r']:
                    if key in self._subform._vars:
                        if key == 'ksize':
                            self._subform._vars[key].set(str(self.ksize.get()))
                        elif key in ['show_original', 'draw_features']:
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
            if 'block_size' in data:
                self.block_size.set(max(2, min(10, int(data['block_size']))))
            if 'ksize' in data:
                val = int(data['ksize'])
                if val <= 3:
                    val = 3
                elif val <= 5:
                    val = 5
                else:
                    val = 7
                self.ksize.set(val)
            if 'k' in data:
                self.k.set(max(0.01, min(0.10, float(data['k']))))
            if 'threshold' in data:
                self.threshold.set(max(0.001, min(0.1, float(data['threshold']))))
            if 'marker_size' in data:
                self.marker_size.set(max(1, min(15, int(data['marker_size']))))
            if 'corner_color_b' in data:
                self.corner_color_b.set(max(0, min(255, int(data['corner_color_b']))))
            if 'corner_color_g' in data:
                self.corner_color_g.set(max(0, min(255, int(data['corner_color_g']))))
            if 'corner_color_r' in data:
                self.corner_color_r.set(max(0, min(255, int(data['corner_color_r']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in ['show_original', 'draw_features', 'block_size', 'ksize', 'k', 'threshold', 'marker_size', 'corner_color_b', 'corner_color_g', 'corner_color_r']:
                    if key in self._subform._vars:
                        if key == 'ksize':
                            self._subform._vars[key].set(str(self.ksize.get()))
                        elif key in ['show_original', 'draw_features']:
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
        lines.append(f"Block Size: {self.block_size.get()}")
        lines.append(f"Sobel Aperture: {self.ksize.get()}")
        lines.append(f"Harris k: {self.k.get():.2f}")
        lines.append(f"Threshold: {self.threshold.get():.3f}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and mark corners on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        gray = np.float32(gray)

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        block_size = self.block_size.get()
        ksize = self.ksize.get()
        k = self.k.get()
        threshold = self.threshold.get()

        # Get color values with error handling for empty spinbox
        try:
            color_b = self.corner_color_b.get()
        except Exception:
            color_b = 0
        try:
            color_g = self.corner_color_g.get()
        except Exception:
            color_g = 0
        try:
            color_r = self.corner_color_r.get()
        except Exception:
            color_r = 255
        color = (color_b, color_g, color_r)

        marker_size = self.marker_size.get()

        # Apply Harris corner detection
        harris = cv2.cornerHarris(gray, block_size, ksize, k)

        # Dilate to mark corners
        harris = cv2.dilate(harris, None)

        # Threshold and find corners
        corner_threshold = threshold * harris.max()
        corners = np.where(harris > corner_threshold)

        # Store raw values for pipeline use
        self.detected_corners = corners

        corners_count = len(corners[0])

        # Draw corners
        if self.draw_features.get():
            for y, x in zip(corners[0], corners[1]):
                cv2.circle(result, (x, y), marker_size, color, -1)

        return result

"""
Shi-Tomasi corner detection effect using OpenCV.

Detects corners using the Shi-Tomasi (goodFeaturesToTrack) algorithm.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ShiTomasiCornersEffect(BaseUIEffect):
    """Detect and mark corners using Shi-Tomasi algorithm"""

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

        # goodFeaturesToTrack parameters
        self.max_corners = tk.IntVar(value=100)
        self.quality_level = tk.DoubleVar(value=0.01)
        self.min_distance = tk.IntVar(value=10)
        self.block_size = tk.IntVar(value=3)

        # Drawing options
        self.corner_color_b = tk.IntVar(value=0)
        self.corner_color_g = tk.IntVar(value=255)
        self.corner_color_r = tk.IntVar(value=0)
        self.marker_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Detect Corners Shi-Tomasi"

    @classmethod
    def get_description(cls) -> str:
        return "Detect corners using Shi-Tomasi (goodFeaturesToTrack)"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Show Original', 'key': 'show_original', 'default': True},
            {'type': 'checkbox', 'label': 'Draw Features', 'key': 'draw_features', 'default': True},
            {'type': 'slider', 'label': 'Max Corners', 'key': 'max_corners', 'min': 1, 'max': 500, 'default': 100},
            {'type': 'slider', 'label': 'Quality Level', 'key': 'quality_level', 'min': 0.001, 'max': 0.1, 'default': 0.01, 'resolution': 0.001},
            {'type': 'slider', 'label': 'Min Distance', 'key': 'min_distance', 'min': 1, 'max': 100, 'default': 10},
            {'type': 'slider', 'label': 'Block Size', 'key': 'block_size', 'min': 3, 'max': 15, 'default': 3},
            {'type': 'slider', 'label': 'Marker Size', 'key': 'marker_size', 'min': 1, 'max': 15, 'default': 5},
            {'type': 'slider', 'label': 'Color B', 'key': 'corner_color_b', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color G', 'key': 'corner_color_g', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'slider', 'label': 'Color R', 'key': 'corner_color_r', 'min': 0, 'max': 255, 'default': 0},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'max_corners': self.max_corners.get(),
            'quality_level': self.quality_level.get(),
            'min_distance': self.min_distance.get(),
            'block_size': self.block_size.get(),
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
            elif key == 'max_corners':
                var.trace_add('write', lambda *args: self.max_corners.set(int(self._subform._vars['max_corners'].get())))
            elif key == 'quality_level':
                var.trace_add('write', lambda *args: self.quality_level.set(float(self._subform._vars['quality_level'].get())))
            elif key == 'min_distance':
                var.trace_add('write', lambda *args: self.min_distance.set(int(self._subform._vars['min_distance'].get())))
            elif key == 'block_size':
                var.trace_add('write', lambda *args: self.block_size.set(int(self._subform._vars['block_size'].get())))
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
        lines.append(f"Max Corners: {self.max_corners.get()}")
        lines.append(f"Quality Level: {self.quality_level.get():.3f}")
        lines.append(f"Min Distance: {self.min_distance.get()}")
        lines.append(f"Block Size: {self.block_size.get()}")
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
            'max_corners': self.max_corners.get(),
            'quality_level': self.quality_level.get(),
            'min_distance': self.min_distance.get(),
            'block_size': self.block_size.get(),
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
                    elif 'max corners' in key:
                        self.max_corners.set(max(1, min(500, int(value))))
                    elif 'quality level' in key:
                        self.quality_level.set(max(0.001, min(0.1, float(value))))
                    elif 'min distance' in key:
                        self.min_distance.set(max(1, min(100, int(value))))
                    elif 'block size' in key:
                        self.block_size.set(max(3, min(15, int(value))))
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
                for key in ['show_original', 'draw_features', 'max_corners', 'quality_level', 'min_distance', 'block_size', 'marker_size', 'corner_color_b', 'corner_color_g', 'corner_color_r']:
                    if key in self._subform._vars:
                        if key in ['show_original', 'draw_features']:
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
            if 'max_corners' in data:
                self.max_corners.set(max(1, min(500, int(data['max_corners']))))
            if 'quality_level' in data:
                self.quality_level.set(max(0.001, min(0.1, float(data['quality_level']))))
            if 'min_distance' in data:
                self.min_distance.set(max(1, min(100, int(data['min_distance']))))
            if 'block_size' in data:
                self.block_size.set(max(3, min(15, int(data['block_size']))))
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
                for key in ['show_original', 'draw_features', 'max_corners', 'quality_level', 'min_distance', 'block_size', 'marker_size', 'corner_color_b', 'corner_color_g', 'corner_color_r']:
                    if key in self._subform._vars:
                        if key in ['show_original', 'draw_features']:
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
        lines.append(f"Max Corners: {self.max_corners.get()}")
        lines.append(f"Quality Level: {self.quality_level.get():.3f}")
        lines.append(f"Min Distance: {self.min_distance.get()}")
        lines.append(f"Block Size: {self.block_size.get()}")
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

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Get parameters
        max_corners = self.max_corners.get()
        quality_level = self.quality_level.get()
        min_distance = self.min_distance.get()
        block_size = self.block_size.get()

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
            color_r = 0
        color = (color_b, color_g, color_r)

        marker_size = self.marker_size.get()

        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            max_corners,
            quality_level,
            min_distance,
            blockSize=block_size
        )

        # Store raw values for pipeline use
        self.detected_corners = corners

        corners_count = 0

        if corners is not None:
            corners_count = len(corners)

            # Draw corners
            if self.draw_features.get():
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(result, (int(x), int(y)), marker_size, color, -1)

        return result

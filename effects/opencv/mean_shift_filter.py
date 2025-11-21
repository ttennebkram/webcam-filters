"""
Pyramid Mean Shift Filtering effect using OpenCV.

Performs color segmentation producing a posterization-like effect.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class MeanShiftFilterEffect(BaseUIEffect):
    """Apply pyramid mean shift filtering for color segmentation"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Mean shift parameters
        self.spatial_radius = tk.IntVar(value=20)  # Spatial window radius
        self.color_radius = tk.IntVar(value=40)  # Color window radius
        self.max_level = tk.IntVar(value=1)  # Max pyramid level

    @classmethod
    def get_name(cls) -> str:
        return "Mean Shift Filter"

    @classmethod
    def get_description(cls) -> str:
        return "Color segmentation via pyramid mean shift"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.pyrMeanShiftFiltering(src, sp, sr)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Spatial Radius', 'key': 'spatial_radius', 'min': 1, 'max': 100, 'default': 20},
            {'type': 'slider', 'label': 'Color Radius', 'key': 'color_radius', 'min': 1, 'max': 100, 'default': 40},
            {'type': 'slider', 'label': 'Max Level', 'key': 'max_level', 'min': 0, 'max': 4, 'default': 1},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'spatial_radius': self.spatial_radius.get(),
            'color_radius': self.color_radius.get(),
            'max_level': self.max_level.get()
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
            if key == 'spatial_radius':
                var.trace_add('write', lambda *args: self.spatial_radius.set(int(self._subform._vars['spatial_radius'].get())))
            elif key == 'color_radius':
                var.trace_add('write', lambda *args: self.color_radius.set(int(self._subform._vars['color_radius'].get())))
            elif key == 'max_level':
                var.trace_add('write', lambda *args: self.max_level.set(int(self._subform._vars['max_level'].get())))

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
        lines.append(f"Spatial Radius: {self.spatial_radius.get()}")
        lines.append(f"Color Radius: {self.color_radius.get()}")
        lines.append(f"Max Level: {self.max_level.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'spatial_radius': self.spatial_radius.get(),
            'color_radius': self.color_radius.get(),
            'max_level': self.max_level.get()
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

                    if 'spatial' in key:
                        self.spatial_radius.set(max(1, min(100, int(value))))
                    elif 'color' in key:
                        self.color_radius.set(max(1, min(100, int(value))))
                    elif 'level' in key or 'max' in key:
                        self.max_level.set(max(0, min(4, int(value))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'spatial_radius' in self._subform._vars:
                    self._subform._vars['spatial_radius'].set(self.spatial_radius.get())
                if 'color_radius' in self._subform._vars:
                    self._subform._vars['color_radius'].set(self.color_radius.get())
                if 'max_level' in self._subform._vars:
                    self._subform._vars['max_level'].set(self.max_level.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'spatial_radius' in data:
                self.spatial_radius.set(max(1, min(100, int(data['spatial_radius']))))
            if 'color_radius' in data:
                self.color_radius.set(max(1, min(100, int(data['color_radius']))))
            if 'max_level' in data:
                self.max_level.set(max(0, min(4, int(data['max_level']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'spatial_radius' in self._subform._vars:
                    self._subform._vars['spatial_radius'].set(self.spatial_radius.get())
                if 'color_radius' in self._subform._vars:
                    self._subform._vars['color_radius'].set(self.color_radius.get())
                if 'max_level' in self._subform._vars:
                    self._subform._vars['max_level'].set(self.max_level.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Spatial Radius: {self.spatial_radius.get()}")
        lines.append(f"Color Radius: {self.color_radius.get()}")
        lines.append(f"Max Level: {self.max_level.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply pyramid mean shift filtering to the frame"""
        if not self.enabled.get():
            return frame

        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Get parameters
        sp = self.spatial_radius.get()
        sr = self.color_radius.get()
        max_level = self.max_level.get()

        # Apply mean shift filtering
        result = cv2.pyrMeanShiftFiltering(frame, sp, sr, maxLevel=max_level)

        return result

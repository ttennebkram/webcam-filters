"""
Bilateral filter effect using OpenCV.

Edge-preserving smoothing that reduces noise while keeping edges sharp.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class BilateralFilterEffect(BaseUIEffect):
    """Apply bilateral filtering for edge-preserving smoothing"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Bilateral filter parameters
        self.diameter = tk.IntVar(value=9)  # Pixel neighborhood diameter
        self.sigma_color = tk.IntVar(value=75)  # Filter sigma in color space
        self.sigma_space = tk.IntVar(value=75)  # Filter sigma in coordinate space

    @classmethod
    def get_name(cls) -> str:
        return "Bilateral Filter"

    @classmethod
    def get_description(cls) -> str:
        return "Edge-preserving smoothing filter"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Diameter', 'key': 'diameter', 'min': 1, 'max': 25, 'default': 9},
            {'type': 'slider', 'label': 'Sigma Color', 'key': 'sigma_color', 'min': 1, 'max': 200, 'default': 75},
            {'type': 'slider', 'label': 'Sigma Space', 'key': 'sigma_space', 'min': 1, 'max': 200, 'default': 75},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'diameter': self.diameter.get(),
            'sigma_color': self.sigma_color.get(),
            'sigma_space': self.sigma_space.get()
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
            if key == 'diameter':
                var.trace_add('write', lambda *args: self.diameter.set(int(self._subform._vars['diameter'].get())))
            elif key == 'sigma_color':
                var.trace_add('write', lambda *args: self.sigma_color.set(int(self._subform._vars['sigma_color'].get())))
            elif key == 'sigma_space':
                var.trace_add('write', lambda *args: self.sigma_space.set(int(self._subform._vars['sigma_space'].get())))

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
        lines.append(f"Diameter: {self.diameter.get()}")
        lines.append(f"Sigma Color: {self.sigma_color.get()}")
        lines.append(f"Sigma Space: {self.sigma_space.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'diameter': self.diameter.get(),
            'sigma_color': self.sigma_color.get(),
            'sigma_space': self.sigma_space.get()
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

                    if 'diameter' in key:
                        self.diameter.set(max(1, min(25, int(value))))
                    elif 'sigma color' in key or 'sigmacolor' in key:
                        self.sigma_color.set(max(1, min(200, int(value))))
                    elif 'sigma space' in key or 'sigmaspace' in key:
                        self.sigma_space.set(max(1, min(200, int(value))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'diameter' in self._subform._vars:
                    self._subform._vars['diameter'].set(self.diameter.get())
                if 'sigma_color' in self._subform._vars:
                    self._subform._vars['sigma_color'].set(self.sigma_color.get())
                if 'sigma_space' in self._subform._vars:
                    self._subform._vars['sigma_space'].set(self.sigma_space.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'diameter' in data:
                self.diameter.set(max(1, min(25, int(data['diameter']))))
            if 'sigma_color' in data:
                self.sigma_color.set(max(1, min(200, int(data['sigma_color']))))
            if 'sigma_space' in data:
                self.sigma_space.set(max(1, min(200, int(data['sigma_space']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'diameter' in self._subform._vars:
                    self._subform._vars['diameter'].set(self.diameter.get())
                if 'sigma_color' in self._subform._vars:
                    self._subform._vars['sigma_color'].set(self.sigma_color.get())
                if 'sigma_space' in self._subform._vars:
                    self._subform._vars['sigma_space'].set(self.sigma_space.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Diameter: {self.diameter.get()}")
        lines.append(f"Sigma Color: {self.sigma_color.get()}")
        lines.append(f"Sigma Space: {self.sigma_space.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply bilateral filter to the frame"""
        if not self.enabled.get():
            return frame

        # Get parameters
        diameter = self.diameter.get()
        sigma_color = self.sigma_color.get()
        sigma_space = self.sigma_space.get()

        # Apply bilateral filter
        result = cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)

        return result

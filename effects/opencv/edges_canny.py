"""
Canny edge detection effect using OpenCV.

Applies Canny edge detection algorithm to find edges in images.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class CannyEffect(BaseUIEffect):
    """Apply Canny edge detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.threshold1 = tk.IntVar(value=30)  # Lower threshold
        self.threshold2 = tk.IntVar(value=150)  # Upper threshold
        self.aperture_size = tk.IntVar(value=3)  # Sobel aperture size
        self.l2_gradient = tk.BooleanVar(value=False)  # Use L2 norm for gradient

    @classmethod
    def get_name(cls) -> str:
        return "Edges Canny"

    @classmethod
    def get_description(cls) -> str:
        return "Apply Canny edge detection algorithm"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.Canny(image, threshold1, threshold2)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Threshold 1 (lower)', 'key': 'threshold1', 'min': 0, 'max': 500, 'default': 30},
            {'type': 'slider', 'label': 'Threshold 2 (upper)', 'key': 'threshold2', 'min': 0, 'max': 500, 'default': 150},
            {'type': 'dropdown', 'label': 'Aperture Size', 'key': 'aperture_size', 'options': [3, 5, 7], 'default': 3},
            {'type': 'checkbox', 'label': 'L2 Gradient', 'key': 'l2_gradient', 'default': False},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'threshold1': self.threshold1.get(),
            'threshold2': self.threshold2.get(),
            'aperture_size': self.aperture_size.get(),
            'l2_gradient': self.l2_gradient.get()
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
            if key == 'threshold1':
                var.trace_add('write', lambda *args: self.threshold1.set(int(self._subform._vars['threshold1'].get())))
            elif key == 'threshold2':
                var.trace_add('write', lambda *args: self.threshold2.set(int(self._subform._vars['threshold2'].get())))
            elif key == 'aperture_size':
                var.trace_add('write', lambda *args: self.aperture_size.set(int(self._subform._vars['aperture_size'].get())))
            elif key == 'l2_gradient':
                var.trace_add('write', lambda *args: self.l2_gradient.set(self._subform._vars['l2_gradient'].get()))

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
        lines.append(f"Threshold 1: {self.threshold1.get()}")
        lines.append(f"Threshold 2: {self.threshold2.get()}")
        lines.append(f"Aperture Size: {self.aperture_size.get()}")
        lines.append(f"L2 Gradient: {'Yes' if self.l2_gradient.get() else 'No'}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'threshold1': self.threshold1.get(),
            'threshold2': self.threshold2.get(),
            'aperture_size': self.aperture_size.get(),
            'l2_gradient': self.l2_gradient.get()
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

                    if 'threshold 1' in key:
                        self.threshold1.set(max(0, min(500, int(value))))
                    elif 'threshold 2' in key:
                        self.threshold2.set(max(0, min(500, int(value))))
                    elif 'aperture' in key:
                        val = int(value)
                        # Clamp to valid aperture sizes
                        if val <= 3:
                            val = 3
                        elif val <= 5:
                            val = 5
                        else:
                            val = 7
                        self.aperture_size.set(val)
                    elif 'l2' in key or 'gradient' in key:
                        self.l2_gradient.set(value.lower() in ('yes', 'true', '1'))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'threshold1' in self._subform._vars:
                    self._subform._vars['threshold1'].set(self.threshold1.get())
                if 'threshold2' in self._subform._vars:
                    self._subform._vars['threshold2'].set(self.threshold2.get())
                if 'aperture_size' in self._subform._vars:
                    self._subform._vars['aperture_size'].set(str(self.aperture_size.get()))
                if 'l2_gradient' in self._subform._vars:
                    self._subform._vars['l2_gradient'].set(self.l2_gradient.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'threshold1' in data:
                self.threshold1.set(max(0, min(500, int(data['threshold1']))))
            if 'threshold2' in data:
                self.threshold2.set(max(0, min(500, int(data['threshold2']))))
            if 'aperture_size' in data:
                val = int(data['aperture_size'])
                # Clamp to valid aperture sizes
                if val <= 3:
                    val = 3
                elif val <= 5:
                    val = 5
                else:
                    val = 7
                self.aperture_size.set(val)
            if 'l2_gradient' in data:
                self.l2_gradient.set(bool(data['l2_gradient']))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'threshold1' in self._subform._vars:
                    self._subform._vars['threshold1'].set(self.threshold1.get())
                if 'threshold2' in self._subform._vars:
                    self._subform._vars['threshold2'].set(self.threshold2.get())
                if 'aperture_size' in self._subform._vars:
                    self._subform._vars['aperture_size'].set(str(self.aperture_size.get()))
                if 'l2_gradient' in self._subform._vars:
                    self._subform._vars['l2_gradient'].set(self.l2_gradient.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Threshold 1: {self.threshold1.get()}")
        lines.append(f"Threshold 2: {self.threshold2.get()}")
        lines.append(f"Aperture Size: {self.aperture_size.get()}")
        lines.append(f"L2 Gradient: {'Yes' if self.l2_gradient.get() else 'No'}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Canny edge detection to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Get parameters
        threshold1 = self.threshold1.get()
        threshold2 = self.threshold2.get()
        aperture_size = self.aperture_size.get()
        l2_gradient = self.l2_gradient.get()

        # Apply Canny edge detection
        edges = cv2.Canny(
            gray,
            threshold1,
            threshold2,
            apertureSize=aperture_size,
            L2gradient=l2_gradient
        )

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return result

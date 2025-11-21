"""
Laplacian edge detection effect using OpenCV.

Second derivative edge detection that highlights regions of rapid intensity change.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class LaplacianEffect(BaseUIEffect):
    """Apply Laplacian edge detection"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Kernel size (must be odd)
        self.ksize = tk.IntVar(value=3)

        # Scale and delta
        self.scale = tk.DoubleVar(value=1.0)
        self.delta = tk.IntVar(value=0)

        # Output options
        self.use_absolute = tk.BooleanVar(value=True)

    @classmethod
    def get_name(cls) -> str:
        return "Edges Laplacian"

    @classmethod
    def get_description(cls) -> str:
        return "Laplacian edge detection (second derivative)"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.Laplacian(src, ddepth)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'dropdown', 'label': 'Kernel Size', 'key': 'ksize', 'options': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], 'default': 3},
            {'type': 'slider', 'label': 'Scale', 'key': 'scale', 'min': 0.1, 'max': 5.0, 'default': 1.0, 'step': 0.1},
            {'type': 'slider', 'label': 'Delta', 'key': 'delta', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'checkbox', 'label': 'Use Absolute Value', 'key': 'use_absolute', 'default': True},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'ksize': self.ksize.get(),
            'scale': self.scale.get(),
            'delta': self.delta.get(),
            'use_absolute': self.use_absolute.get()
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
            if key == 'ksize':
                var.trace_add('write', lambda *args: self.ksize.set(int(self._subform._vars['ksize'].get())))
            elif key == 'scale':
                var.trace_add('write', lambda *args: self.scale.set(float(self._subform._vars['scale'].get())))
            elif key == 'delta':
                var.trace_add('write', lambda *args: self.delta.set(int(self._subform._vars['delta'].get())))
            elif key == 'use_absolute':
                var.trace_add('write', lambda *args: self.use_absolute.set(self._subform._vars['use_absolute'].get()))

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
        lines.append(f"Kernel Size: {self.ksize.get()}")
        lines.append(f"Scale: {self.scale.get():.1f}")
        lines.append(f"Delta: {self.delta.get()}")
        lines.append(f"Use Absolute Value: {'Yes' if self.use_absolute.get() else 'No'}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'ksize': self.ksize.get(),
            'scale': self.scale.get(),
            'delta': self.delta.get(),
            'use_absolute': self.use_absolute.get()
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

                    if 'kernel' in key:
                        val = int(value)
                        # Ensure odd value
                        if val % 2 == 0:
                            val += 1
                        self.ksize.set(max(1, min(31, val)))
                    elif 'scale' in key:
                        self.scale.set(max(0.1, min(5.0, float(value))))
                    elif 'delta' in key:
                        self.delta.set(max(0, min(255, int(value))))
                    elif 'absolute' in key:
                        self.use_absolute.set(value.lower() in ('yes', 'true', '1'))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'ksize' in self._subform._vars:
                    self._subform._vars['ksize'].set(str(self.ksize.get()))
                if 'scale' in self._subform._vars:
                    self._subform._vars['scale'].set(self.scale.get())
                if 'delta' in self._subform._vars:
                    self._subform._vars['delta'].set(self.delta.get())
                if 'use_absolute' in self._subform._vars:
                    self._subform._vars['use_absolute'].set(self.use_absolute.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'ksize' in data:
                val = int(data['ksize'])
                # Ensure odd value
                if val % 2 == 0:
                    val += 1
                self.ksize.set(max(1, min(31, val)))
            if 'scale' in data:
                self.scale.set(max(0.1, min(5.0, float(data['scale']))))
            if 'delta' in data:
                self.delta.set(max(0, min(255, int(data['delta']))))
            if 'use_absolute' in data:
                self.use_absolute.set(bool(data['use_absolute']))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'ksize' in self._subform._vars:
                    self._subform._vars['ksize'].set(str(self.ksize.get()))
                if 'scale' in self._subform._vars:
                    self._subform._vars['scale'].set(self.scale.get())
                if 'delta' in self._subform._vars:
                    self._subform._vars['delta'].set(self.delta.get())
                if 'use_absolute' in self._subform._vars:
                    self._subform._vars['use_absolute'].set(self.use_absolute.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Kernel Size: {self.ksize.get()}")
        lines.append(f"Scale: {self.scale.get():.1f}")
        lines.append(f"Delta: {self.delta.get()}")
        lines.append(f"Use Absolute Value: {'Yes' if self.use_absolute.get() else 'No'}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Laplacian edge detection to the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Get parameters
        ksize = self.ksize.get()
        if ksize % 2 == 0:
            ksize += 1
        scale = self.scale.get()
        delta = self.delta.get()

        # Apply Laplacian
        result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)

        # Convert to absolute and 8-bit
        if self.use_absolute.get():
            result = np.absolute(result)

        result = np.uint8(np.clip(result, 0, 255))

        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

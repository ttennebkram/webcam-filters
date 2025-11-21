"""
Blur effect using OpenCV.

Applies Gaussian blur to the image with adjustable parameters.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class BlurEffect(BaseUIEffect):
    """Apply Gaussian blur effect to the image"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.kernel_size_x = tk.IntVar(value=7)  # Kernel width
        self.kernel_size_y = tk.IntVar(value=7)  # Kernel height
        self.sigma_x = tk.DoubleVar(value=0.0)  # 0 means calculated from kernel size

    @classmethod
    def get_name(cls) -> str:
        return "Blur"

    @classmethod
    def get_description(cls) -> str:
        return "Apply Gaussian blur effect to soften the image"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.GaussianBlur(src, ksize, sigmaX)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Kernel Size X', 'key': 'kernel_size_x', 'min': 1, 'max': 31, 'default': 7},
            {'type': 'slider', 'label': 'Kernel Size Y', 'key': 'kernel_size_y', 'min': 1, 'max': 31, 'default': 7},
            {'type': 'slider', 'label': 'Sigma X', 'key': 'sigma_x', 'min': 0, 'max': 10, 'default': 0, 'resolution': 0.1},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'kernel_size_x': self.kernel_size_x.get(),
            'kernel_size_y': self.kernel_size_y.get(),
            'sigma_x': self.sigma_x.get()
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
            if key == 'kernel_size_x':
                var.trace_add('write', lambda *args: self._sync_kernel_x())
            elif key == 'kernel_size_y':
                var.trace_add('write', lambda *args: self._sync_kernel_y())
            elif key == 'sigma_x':
                var.trace_add('write', lambda *args: self.sigma_x.set(float(self._subform._vars['sigma_x'].get())))

    def _sync_kernel_x(self):
        """Sync kernel size X ensuring odd value"""
        ksize = int(self._subform._vars['kernel_size_x'].get())
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size_x.set(ksize)

    def _sync_kernel_y(self):
        """Sync kernel size Y ensuring odd value"""
        ksize = int(self._subform._vars['kernel_size_y'].get())
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size_y.set(ksize)

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
        lines.append(f"Kernel Size X: {self.kernel_size_x.get()}")
        lines.append(f"Kernel Size Y: {self.kernel_size_y.get()}")
        sigma = self.sigma_x.get()
        if sigma == 0:
            lines.append(f"Sigma X: {sigma} (auto)")
        else:
            lines.append(f"Sigma X: {sigma}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'kernel_size_x': self.kernel_size_x.get(),
            'kernel_size_y': self.kernel_size_y.get(),
            'sigma_x': self.sigma_x.get()
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

                    if 'kernel size x' in key:
                        ksize = max(1, min(31, int(value)))
                        if ksize % 2 == 0:
                            ksize += 1
                        self.kernel_size_x.set(ksize)
                    elif 'kernel size y' in key:
                        ksize = max(1, min(31, int(value)))
                        if ksize % 2 == 0:
                            ksize += 1
                        self.kernel_size_y.set(ksize)
                    elif 'sigma' in key:
                        # Handle "0 (auto)" format
                        value = value.split('(')[0].strip()
                        self.sigma_x.set(max(0, min(10, float(value))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size_x' in self._subform._vars:
                    self._subform._vars['kernel_size_x'].set(self.kernel_size_x.get())
                if 'kernel_size_y' in self._subform._vars:
                    self._subform._vars['kernel_size_y'].set(self.kernel_size_y.get())
                if 'sigma_x' in self._subform._vars:
                    self._subform._vars['sigma_x'].set(self.sigma_x.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'kernel_size_x' in data:
                ksize = max(1, min(31, int(data['kernel_size_x'])))
                if ksize % 2 == 0:
                    ksize += 1
                self.kernel_size_x.set(ksize)
            if 'kernel_size_y' in data:
                ksize = max(1, min(31, int(data['kernel_size_y'])))
                if ksize % 2 == 0:
                    ksize += 1
                self.kernel_size_y.set(ksize)
            if 'sigma_x' in data:
                self.sigma_x.set(max(0, min(10, float(data['sigma_x']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size_x' in self._subform._vars:
                    self._subform._vars['kernel_size_x'].set(self.kernel_size_x.get())
                if 'kernel_size_y' in self._subform._vars:
                    self._subform._vars['kernel_size_y'].set(self.kernel_size_y.get())
                if 'sigma_x' in self._subform._vars:
                    self._subform._vars['sigma_x'].set(self.sigma_x.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Kernel Size X: {self.kernel_size_x.get()}")
        lines.append(f"Kernel Size Y: {self.kernel_size_y.get()}")
        sigma = self.sigma_x.get()
        if sigma == 0:
            lines.append(f"Sigma X: {sigma} (auto)")
        else:
            lines.append(f"Sigma X: {sigma}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Gaussian blur to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get kernel sizes (ensure they're odd)
        ksize_x = self.kernel_size_x.get()
        if ksize_x % 2 == 0:
            ksize_x = ksize_x + 1

        ksize_y = self.kernel_size_y.get()
        if ksize_y % 2 == 0:
            ksize_y = ksize_y + 1

        # Get sigma
        sigma_x = self.sigma_x.get()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (ksize_x, ksize_y), sigma_x)

        return blurred

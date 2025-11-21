"""
Median blur effect using OpenCV.

Noise reduction filter that replaces each pixel with the median of neighboring pixels.
Particularly effective at removing salt-and-pepper noise while preserving edges.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class MedianBlurEffect(BaseUIEffect):
    """Apply median blur for noise reduction"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Median blur parameter (must be odd)
        self.kernel_size = tk.IntVar(value=5)

    @classmethod
    def get_name(cls) -> str:
        return "Median Blur"

    @classmethod
    def get_description(cls) -> str:
        return "Noise reduction using median filter"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.medianBlur(src, ksize)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Kernel Size', 'key': 'kernel_size', 'min': 1, 'max': 31, 'default': 5},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'kernel_size': self.kernel_size.get()
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
            if key == 'kernel_size':
                var.trace_add('write', lambda *args: self._sync_kernel_size())

    def _sync_kernel_size(self):
        """Sync kernel size ensuring odd value"""
        ksize = int(self._subform._vars['kernel_size'].get())
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size.set(ksize)

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
        lines.append(f"Kernel Size: {self.kernel_size.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'kernel_size': self.kernel_size.get()
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
                        ksize = max(1, min(31, int(value)))
                        if ksize % 2 == 0:
                            ksize += 1
                        self.kernel_size.set(ksize)

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size' in self._subform._vars:
                    self._subform._vars['kernel_size'].set(self.kernel_size.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'kernel_size' in data:
                ksize = max(1, min(31, int(data['kernel_size'])))
                if ksize % 2 == 0:
                    ksize += 1
                self.kernel_size.set(ksize)

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size' in self._subform._vars:
                    self._subform._vars['kernel_size'].set(self.kernel_size.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        return f"Kernel Size: {self.kernel_size.get()}"

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply median blur to the frame"""
        if not self.enabled.get():
            return frame

        # Get kernel size (must be odd)
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1

        # Apply median blur
        result = cv2.medianBlur(frame, ksize)

        return result

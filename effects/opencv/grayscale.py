"""
Color conversion effect using OpenCV.

Converts images between different color spaces using cv2.cvtColor.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class GrayscaleEffect(BaseUIEffect):
    """Convert image to different color spaces"""

    # Color conversion options (code, display name)
    COLOR_CONVERSIONS = [
        (cv2.COLOR_BGR2GRAY, "BGR to Grayscale"),
        (cv2.COLOR_BGR2RGB, "BGR to RGB"),
        (cv2.COLOR_BGR2HSV, "BGR to HSV"),
        (cv2.COLOR_BGR2HLS, "BGR to HLS"),
        (cv2.COLOR_BGR2LAB, "BGR to LAB"),
        (cv2.COLOR_BGR2LUV, "BGR to LUV"),
        (cv2.COLOR_BGR2YCrCb, "BGR to YCrCb"),
        (cv2.COLOR_BGR2XYZ, "BGR to XYZ"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.conversion_index = tk.IntVar(value=0)  # Index into COLOR_CONVERSIONS

    @classmethod
    def get_name(cls) -> str:
        return "Grayscale / Color Conversion"

    @classmethod
    def get_description(cls) -> str:
        return "Convert image between different color spaces"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.cvtColor(src, code)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        conversion_names = [name for _, name in self.COLOR_CONVERSIONS]
        return [
            {'type': 'dropdown', 'label': 'Conversion', 'key': 'conversion', 'options': conversion_names, 'default': 'BGR to Grayscale'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        conv_idx = self.conversion_index.get()
        conv_name = self.COLOR_CONVERSIONS[conv_idx][1] if conv_idx < len(self.COLOR_CONVERSIONS) else "BGR to Grayscale"
        return {
            'conversion': conv_name
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
        if 'conversion' in self._subform._vars:
            def on_conversion_change(*args):
                conv_name = self._subform._vars['conversion'].get()
                for i, (_, name) in enumerate(self.COLOR_CONVERSIONS):
                    if name == conv_name:
                        self.conversion_index.set(i)
                        break
            self._subform._vars['conversion'].trace_add('write', on_conversion_change)

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
        conv_idx = self.conversion_index.get()
        conv_name = self.COLOR_CONVERSIONS[conv_idx][1] if conv_idx < len(self.COLOR_CONVERSIONS) else "Unknown"
        lines.append(f"Conversion: {conv_name}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        conv_idx = self.conversion_index.get()
        conv_name = self.COLOR_CONVERSIONS[conv_idx][1] if conv_idx < len(self.COLOR_CONVERSIONS) else "BGR to Grayscale"
        data = {
            'effect': self.get_name(),
            'conversion': conv_name,
            'conversion_index': conv_idx
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

                    if 'conversion' in key:
                        # Find matching conversion
                        for i, (_, name) in enumerate(self.COLOR_CONVERSIONS):
                            if name == value:
                                self.conversion_index.set(i)
                                break

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'conversion' in self._subform._vars:
                    conv_idx = self.conversion_index.get()
                    conv_name = self.COLOR_CONVERSIONS[conv_idx][1]
                    self._subform._vars['conversion'].set(conv_name)
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'conversion_index' in data:
                idx = max(0, min(len(self.COLOR_CONVERSIONS) - 1, int(data['conversion_index'])))
                self.conversion_index.set(idx)
            elif 'conversion' in data:
                # Find matching conversion by name
                for i, (_, name) in enumerate(self.COLOR_CONVERSIONS):
                    if name == data['conversion']:
                        self.conversion_index.set(i)
                        break

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'conversion' in self._subform._vars:
                    conv_idx = self.conversion_index.get()
                    conv_name = self.COLOR_CONVERSIONS[conv_idx][1]
                    self._subform._vars['conversion'].set(conv_name)
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        conv_idx = self.conversion_index.get()
        conv_name = self.COLOR_CONVERSIONS[conv_idx][1] if conv_idx < len(self.COLOR_CONVERSIONS) else "Unknown"
        return f"Conversion: {conv_name}"

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply color conversion to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get selected conversion
        conv_code, _ = self.COLOR_CONVERSIONS[self.conversion_index.get()]

        # Apply conversion
        converted = cv2.cvtColor(frame, conv_code)

        # If result is grayscale, convert back to BGR for display
        if len(converted.shape) == 2:
            converted = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)

        return converted

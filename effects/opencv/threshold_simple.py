"""
Simple threshold effect using OpenCV.

Applies basic thresholding to convert images to binary.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class ThresholdSimpleEffect(BaseUIEffect):
    """Apply simple thresholding to create binary images"""

    # Threshold type options (code, display name)
    THRESHOLD_TYPES = [
        (cv2.THRESH_BINARY, "THRESH_BINARY"),
        (cv2.THRESH_BINARY_INV, "THRESH_BINARY_INV"),
        (cv2.THRESH_TRUNC, "THRESH_TRUNC"),
        (cv2.THRESH_TOZERO, "THRESH_TOZERO"),
        (cv2.THRESH_TOZERO_INV, "THRESH_TOZERO_INV"),
    ]

    # Modifier flags (can be OR'd with basic types)
    THRESHOLD_MODIFIERS = [
        (0, "None"),
        (cv2.THRESH_OTSU, "THRESH_OTSU"),
        (cv2.THRESH_TRIANGLE, "THRESH_TRIANGLE"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.thresh_value = tk.IntVar(value=200)  # Threshold value
        self.max_value = tk.IntVar(value=255)  # Maximum value
        self.thresh_type_index = tk.IntVar(value=0)  # Default to THRESH_BINARY
        self.thresh_modifier_index = tk.IntVar(value=0)  # Default to None

    @classmethod
    def get_name(cls) -> str:
        return "Threshold (Simple)"

    @classmethod
    def get_description(cls) -> str:
        return "Apply simple thresholding to create binary images"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.threshold(src, thresh, maxval, type)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        type_names = [name for _, name in self.THRESHOLD_TYPES]
        modifier_names = [name for _, name in self.THRESHOLD_MODIFIERS]
        return [
            {'type': 'slider', 'label': 'Threshold', 'key': 'thresh_value', 'min': 0, 'max': 255, 'default': 200},
            {'type': 'slider', 'label': 'Max Value', 'key': 'max_value', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'dropdown', 'label': 'Type', 'key': 'thresh_type', 'options': type_names, 'default': 'THRESH_BINARY'},
            {'type': 'dropdown', 'label': 'Modifier', 'key': 'thresh_modifier', 'options': modifier_names, 'default': 'None'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        type_idx = self.thresh_type_index.get()
        type_name = self.THRESHOLD_TYPES[type_idx][1] if type_idx < len(self.THRESHOLD_TYPES) else "THRESH_BINARY"
        modifier_idx = self.thresh_modifier_index.get()
        modifier_name = self.THRESHOLD_MODIFIERS[modifier_idx][1] if modifier_idx < len(self.THRESHOLD_MODIFIERS) else "None"
        return {
            'thresh_value': self.thresh_value.get(),
            'max_value': self.max_value.get(),
            'thresh_type': type_name,
            'thresh_modifier': modifier_name
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
        if 'thresh_value' in self._subform._vars:
            self._subform._vars['thresh_value'].trace_add('write',
                lambda *args: self.thresh_value.set(int(self._subform._vars['thresh_value'].get())))
        if 'max_value' in self._subform._vars:
            self._subform._vars['max_value'].trace_add('write',
                lambda *args: self.max_value.set(int(self._subform._vars['max_value'].get())))
        if 'thresh_type' in self._subform._vars:
            def on_type_change(*args):
                type_name = self._subform._vars['thresh_type'].get()
                for i, (_, name) in enumerate(self.THRESHOLD_TYPES):
                    if name == type_name:
                        self.thresh_type_index.set(i)
                        break
            self._subform._vars['thresh_type'].trace_add('write', on_type_change)
        if 'thresh_modifier' in self._subform._vars:
            def on_modifier_change(*args):
                modifier_name = self._subform._vars['thresh_modifier'].get()
                for i, (_, name) in enumerate(self.THRESHOLD_MODIFIERS):
                    if name == modifier_name:
                        self.thresh_modifier_index.set(i)
                        break
            self._subform._vars['thresh_modifier'].trace_add('write', on_modifier_change)

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
        lines.append(f"Threshold: {self.thresh_value.get()}")
        lines.append(f"Max Value: {self.max_value.get()}")
        type_idx = self.thresh_type_index.get()
        type_name = self.THRESHOLD_TYPES[type_idx][1] if type_idx < len(self.THRESHOLD_TYPES) else "Unknown"
        lines.append(f"Type: {type_name}")
        modifier_idx = self.thresh_modifier_index.get()
        modifier_name = self.THRESHOLD_MODIFIERS[modifier_idx][1] if modifier_idx < len(self.THRESHOLD_MODIFIERS) else "Unknown"
        lines.append(f"Modifier: {modifier_name}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        type_idx = self.thresh_type_index.get()
        type_name = self.THRESHOLD_TYPES[type_idx][1] if type_idx < len(self.THRESHOLD_TYPES) else "THRESH_BINARY"
        modifier_idx = self.thresh_modifier_index.get()
        modifier_name = self.THRESHOLD_MODIFIERS[modifier_idx][1] if modifier_idx < len(self.THRESHOLD_MODIFIERS) else "None"
        data = {
            'effect': self.get_name(),
            'thresh_value': self.thresh_value.get(),
            'max_value': self.max_value.get(),
            'thresh_type': type_name,
            'thresh_type_index': type_idx,
            'thresh_modifier': modifier_name,
            'thresh_modifier_index': modifier_idx
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

                    if key == 'threshold':
                        self.thresh_value.set(max(0, min(255, int(value))))
                    elif 'max' in key:
                        self.max_value.set(max(0, min(255, int(value))))
                    elif key == 'type':
                        for i, (_, name) in enumerate(self.THRESHOLD_TYPES):
                            if name == value:
                                self.thresh_type_index.set(i)
                                break
                    elif 'modifier' in key:
                        for i, (_, name) in enumerate(self.THRESHOLD_MODIFIERS):
                            if name == value:
                                self.thresh_modifier_index.set(i)
                                break

            # Update subform variables if in edit mode
            self._sync_subform_from_effect()
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'thresh_value' in data:
                self.thresh_value.set(max(0, min(255, int(data['thresh_value']))))
            if 'max_value' in data:
                self.max_value.set(max(0, min(255, int(data['max_value']))))
            if 'thresh_type_index' in data:
                idx = max(0, min(len(self.THRESHOLD_TYPES) - 1, int(data['thresh_type_index'])))
                self.thresh_type_index.set(idx)
            elif 'thresh_type' in data:
                for i, (_, name) in enumerate(self.THRESHOLD_TYPES):
                    if name == data['thresh_type']:
                        self.thresh_type_index.set(i)
                        break
            if 'thresh_modifier_index' in data:
                idx = max(0, min(len(self.THRESHOLD_MODIFIERS) - 1, int(data['thresh_modifier_index'])))
                self.thresh_modifier_index.set(idx)
            elif 'thresh_modifier' in data:
                for i, (_, name) in enumerate(self.THRESHOLD_MODIFIERS):
                    if name == data['thresh_modifier']:
                        self.thresh_modifier_index.set(i)
                        break

            # Update subform variables if in edit mode
            self._sync_subform_from_effect()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_effect(self):
        """Sync subform variables from effect variables"""
        if self._current_mode == 'edit' and hasattr(self, '_subform'):
            if 'thresh_value' in self._subform._vars:
                self._subform._vars['thresh_value'].set(self.thresh_value.get())
            if 'max_value' in self._subform._vars:
                self._subform._vars['max_value'].set(self.max_value.get())
            if 'thresh_type' in self._subform._vars:
                type_idx = self.thresh_type_index.get()
                type_name = self.THRESHOLD_TYPES[type_idx][1]
                self._subform._vars['thresh_type'].set(type_name)
            if 'thresh_modifier' in self._subform._vars:
                modifier_idx = self.thresh_modifier_index.get()
                modifier_name = self.THRESHOLD_MODIFIERS[modifier_idx][1]
                self._subform._vars['thresh_modifier'].set(modifier_name)

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Threshold: {self.thresh_value.get()}")
        lines.append(f"Max Value: {self.max_value.get()}")
        type_idx = self.thresh_type_index.get()
        type_name = self.THRESHOLD_TYPES[type_idx][1] if type_idx < len(self.THRESHOLD_TYPES) else "Unknown"
        lines.append(f"Type: {type_name}")
        modifier_idx = self.thresh_modifier_index.get()
        modifier_name = self.THRESHOLD_MODIFIERS[modifier_idx][1] if modifier_idx < len(self.THRESHOLD_MODIFIERS) else "Unknown"
        lines.append(f"Modifier: {modifier_name}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply thresholding to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get parameters
        thresh = self.thresh_value.get()
        maxval = self.max_value.get()
        thresh_type, _ = self.THRESHOLD_TYPES[self.thresh_type_index.get()]
        modifier, _ = self.THRESHOLD_MODIFIERS[self.thresh_modifier_index.get()]

        # Combine type with modifier using bitwise OR
        combined_type = thresh_type | modifier

        # OTSU and TRIANGLE require grayscale input
        if modifier in (cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE):
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Apply threshold
            t_value, thresholded = cv2.threshold(gray, thresh, maxval, combined_type)

            # Convert back to BGR (3 identical channels)
            result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
            return result
        else:
            # Apply threshold (works on grayscale or each BGR channel separately)
            t_value, thresholded = cv2.threshold(frame, thresh, maxval, combined_type)

            return thresholded

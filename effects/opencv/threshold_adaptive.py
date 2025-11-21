"""
Adaptive threshold effect using OpenCV.

Applies adaptive thresholding which calculates threshold for small regions.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ThresholdAdaptiveEffect(BaseUIEffect):
    """Apply adaptive thresholding to create binary images"""

    # Adaptive method options
    ADAPTIVE_METHODS = [
        (cv2.ADAPTIVE_THRESH_MEAN_C, "ADAPTIVE_THRESH_MEAN_C"),
        (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "ADAPTIVE_THRESH_GAUSSIAN_C"),
    ]

    # Threshold type options
    THRESHOLD_TYPES = [
        (cv2.THRESH_BINARY, "THRESH_BINARY"),
        (cv2.THRESH_BINARY_INV, "THRESH_BINARY_INV"),
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.max_value = tk.IntVar(value=255)
        self.adaptive_method_index = tk.IntVar(value=0)  # ADAPTIVE_THRESH_MEAN_C
        self.thresh_type_index = tk.IntVar(value=0)  # THRESH_BINARY
        self.block_size = tk.IntVar(value=25)  # Must be odd
        self.c_value = tk.IntVar(value=15)  # Constant subtracted from mean

    @classmethod
    def get_name(cls) -> str:
        return "Threshold (Adaptive)"

    @classmethod
    def get_description(cls) -> str:
        return "Apply adaptive thresholding with local region calculation"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        method_names = [name for _, name in self.ADAPTIVE_METHODS]
        type_names = [name for _, name in self.THRESHOLD_TYPES]
        # Odd values from 3 to 99
        block_sizes = [i for i in range(3, 100, 2)]

        return [
            {'type': 'slider', 'label': 'Max Value', 'key': 'max_value', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'dropdown', 'label': 'Adaptive Method', 'key': 'adaptive_method_index', 'options': method_names, 'default': 'ADAPTIVE_THRESH_MEAN_C'},
            {'type': 'dropdown', 'label': 'Threshold Type', 'key': 'thresh_type_index', 'options': type_names, 'default': 'THRESH_BINARY'},
            {'type': 'dropdown', 'label': 'Block Size', 'key': 'block_size', 'options': block_sizes, 'default': 25},
            {'type': 'slider', 'label': 'C (constant)', 'key': 'c_value', 'min': -50, 'max': 50, 'default': 15},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        method_names = [name for _, name in self.ADAPTIVE_METHODS]
        type_names = [name for _, name in self.THRESHOLD_TYPES]
        return {
            'max_value': self.max_value.get(),
            'adaptive_method_index': method_names[self.adaptive_method_index.get()],
            'thresh_type_index': type_names[self.thresh_type_index.get()],
            'block_size': self.block_size.get(),
            'c_value': self.c_value.get()
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
        method_names = [name for _, name in self.ADAPTIVE_METHODS]
        type_names = [name for _, name in self.THRESHOLD_TYPES]

        # When subform values change, update effect's tk.Variables
        for key, var in self._subform._vars.items():
            if key == 'max_value':
                var.trace_add('write', lambda *args: self.max_value.set(int(self._subform._vars['max_value'].get())))
            elif key == 'adaptive_method_index':
                def update_method(*args):
                    val = self._subform._vars['adaptive_method_index'].get()
                    if val in method_names:
                        self.adaptive_method_index.set(method_names.index(val))
                var.trace_add('write', update_method)
            elif key == 'thresh_type_index':
                def update_type(*args):
                    val = self._subform._vars['thresh_type_index'].get()
                    if val in type_names:
                        self.thresh_type_index.set(type_names.index(val))
                var.trace_add('write', update_type)
            elif key == 'block_size':
                var.trace_add('write', lambda *args: self.block_size.set(int(self._subform._vars['block_size'].get())))
            elif key == 'c_value':
                var.trace_add('write', lambda *args: self.c_value.set(int(self._subform._vars['c_value'].get())))

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
        lines.append(f"Max Value: {self.max_value.get()}")
        method_name = self.ADAPTIVE_METHODS[self.adaptive_method_index.get()][1]
        lines.append(f"Adaptive Method: {method_name}")
        type_name = self.THRESHOLD_TYPES[self.thresh_type_index.get()][1]
        lines.append(f"Threshold Type: {type_name}")
        lines.append(f"Block Size: {self.block_size.get()}")
        lines.append(f"C: {self.c_value.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'max_value': self.max_value.get(),
            'adaptive_method': self.ADAPTIVE_METHODS[self.adaptive_method_index.get()][1],
            'threshold_type': self.THRESHOLD_TYPES[self.thresh_type_index.get()][1],
            'block_size': self.block_size.get(),
            'c_value': self.c_value.get()
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
            method_names = [name for _, name in self.ADAPTIVE_METHODS]
            type_names = [name for _, name in self.THRESHOLD_TYPES]

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if 'max' in key and 'value' in key:
                        self.max_value.set(max(0, min(255, int(value))))
                    elif 'adaptive' in key and 'method' in key:
                        if value in method_names:
                            self.adaptive_method_index.set(method_names.index(value))
                    elif 'threshold' in key and 'type' in key:
                        if value in type_names:
                            self.thresh_type_index.set(type_names.index(value))
                    elif 'block' in key:
                        val = int(value)
                        if val % 2 == 0:
                            val += 1
                        self.block_size.set(max(3, min(99, val)))
                    elif key == 'c':
                        self.c_value.set(max(-50, min(50, int(value))))

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
            method_names = [name for _, name in self.ADAPTIVE_METHODS]
            type_names = [name for _, name in self.THRESHOLD_TYPES]

            if 'max_value' in data:
                self.max_value.set(max(0, min(255, int(data['max_value']))))
            if 'adaptive_method' in data:
                if data['adaptive_method'] in method_names:
                    self.adaptive_method_index.set(method_names.index(data['adaptive_method']))
            if 'threshold_type' in data:
                if data['threshold_type'] in type_names:
                    self.thresh_type_index.set(type_names.index(data['threshold_type']))
            if 'block_size' in data:
                val = int(data['block_size'])
                if val % 2 == 0:
                    val += 1
                self.block_size.set(max(3, min(99, val)))
            if 'c_value' in data:
                self.c_value.set(max(-50, min(50, int(data['c_value']))))

            # Update subform variables if in edit mode
            self._sync_subform_from_effect()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_effect(self):
        """Sync subform variables from effect variables"""
        if self._current_mode == 'edit' and hasattr(self, '_subform'):
            method_names = [name for _, name in self.ADAPTIVE_METHODS]
            type_names = [name for _, name in self.THRESHOLD_TYPES]

            if 'max_value' in self._subform._vars:
                self._subform._vars['max_value'].set(self.max_value.get())
            if 'adaptive_method_index' in self._subform._vars:
                self._subform._vars['adaptive_method_index'].set(method_names[self.adaptive_method_index.get()])
            if 'thresh_type_index' in self._subform._vars:
                self._subform._vars['thresh_type_index'].set(type_names[self.thresh_type_index.get()])
            if 'block_size' in self._subform._vars:
                self._subform._vars['block_size'].set(str(self.block_size.get()))
            if 'c_value' in self._subform._vars:
                self._subform._vars['c_value'].set(self.c_value.get())

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Max Value: {self.max_value.get()}")
        method_idx = self.adaptive_method_index.get()
        method_name = self.ADAPTIVE_METHODS[method_idx][1] if method_idx < len(self.ADAPTIVE_METHODS) else "Unknown"
        lines.append(f"Adaptive Method: {method_name}")
        type_idx = self.thresh_type_index.get()
        type_name = self.THRESHOLD_TYPES[type_idx][1] if type_idx < len(self.THRESHOLD_TYPES) else "Unknown"
        lines.append(f"Threshold Type: {type_name}")
        block = self.block_size.get()
        if block % 2 == 0:
            block += 1
        lines.append(f"Block Size: {block}")
        lines.append(f"C: {self.c_value.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply adaptive thresholding to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Get parameters
        maxval = self.max_value.get()
        adaptive_method, _ = self.ADAPTIVE_METHODS[self.adaptive_method_index.get()]
        thresh_type, _ = self.THRESHOLD_TYPES[self.thresh_type_index.get()]
        block_size = self.block_size.get()
        c_value = self.c_value.get()

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size = block_size + 1

        # Adaptive threshold requires grayscale input
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply adaptive threshold
        thresholded = cv2.adaptiveThreshold(
            gray, maxval, adaptive_method, thresh_type, block_size, c_value
        )

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        return result

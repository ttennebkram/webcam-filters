"""
Color inRange filtering effect using OpenCV.

Filters pixels by color range in HSV or BGR color space.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class InRangeEffect(BaseUIEffect):
    """Filter colors within a specified range"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Color space
        self.use_hsv = tk.BooleanVar(value=True)

        # HSV ranges (Hue: 0-179, Sat: 0-255, Val: 0-255)
        self.h_low = tk.IntVar(value=0)
        self.h_high = tk.IntVar(value=179)
        self.s_low = tk.IntVar(value=0)
        self.s_high = tk.IntVar(value=255)
        self.v_low = tk.IntVar(value=0)
        self.v_high = tk.IntVar(value=255)

        # Output mode
        self.output_mode = tk.StringVar(value="mask")  # mask, masked, inverse

    @classmethod
    def get_name(cls) -> str:
        return "Color In Range"

    @classmethod
    def get_description(cls) -> str:
        return "Filter colors within a specified range"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.inRange(src, lowerb, upperb)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Use HSV (uncheck for BGR)', 'key': 'use_hsv', 'default': True},
            {'type': 'slider', 'label': 'H/B Low', 'key': 'h_low', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'H/B High', 'key': 'h_high', 'min': 0, 'max': 255, 'default': 179},
            {'type': 'slider', 'label': 'S/G Low', 'key': 's_low', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'S/G High', 'key': 's_high', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'slider', 'label': 'V/R Low', 'key': 'v_low', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'V/R High', 'key': 'v_high', 'min': 0, 'max': 255, 'default': 255},
            {'type': 'dropdown', 'label': 'Output', 'key': 'output_mode', 'options': ['Mask Only', 'Keep In-Range', 'Keep Out-of-Range'], 'default': 'Mask Only'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        output_map = {
            'mask': 'Mask Only',
            'masked': 'Keep In-Range',
            'inverse': 'Keep Out-of-Range'
        }
        return {
            'use_hsv': self.use_hsv.get(),
            'h_low': self.h_low.get(),
            'h_high': self.h_high.get(),
            's_low': self.s_low.get(),
            's_high': self.s_high.get(),
            'v_low': self.v_low.get(),
            'v_high': self.v_high.get(),
            'output_mode': output_map.get(self.output_mode.get(), 'Mask Only')
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
            if key == 'use_hsv':
                var.trace_add('write', lambda *args: self.use_hsv.set(self._subform._vars['use_hsv'].get()))
            elif key == 'h_low':
                var.trace_add('write', lambda *args: self.h_low.set(int(self._subform._vars['h_low'].get())))
            elif key == 'h_high':
                var.trace_add('write', lambda *args: self.h_high.set(int(self._subform._vars['h_high'].get())))
            elif key == 's_low':
                var.trace_add('write', lambda *args: self.s_low.set(int(self._subform._vars['s_low'].get())))
            elif key == 's_high':
                var.trace_add('write', lambda *args: self.s_high.set(int(self._subform._vars['s_high'].get())))
            elif key == 'v_low':
                var.trace_add('write', lambda *args: self.v_low.set(int(self._subform._vars['v_low'].get())))
            elif key == 'v_high':
                var.trace_add('write', lambda *args: self.v_high.set(int(self._subform._vars['v_high'].get())))
            elif key == 'output_mode':
                var.trace_add('write', lambda *args: self._update_output_mode())

    def _update_output_mode(self):
        """Update output mode from subform value"""
        value = self._subform._vars['output_mode'].get()
        output_map = {
            'Mask Only': 'mask',
            'Keep In-Range': 'masked',
            'Keep Out-of-Range': 'inverse'
        }
        self.output_mode.set(output_map.get(value, 'mask'))

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
        lines.append(f"Color Space: {'HSV' if self.use_hsv.get() else 'BGR'}")

        if self.use_hsv.get():
            lines.append(f"H: {self.h_low.get()} - {self.h_high.get()}")
            lines.append(f"S: {self.s_low.get()} - {self.s_high.get()}")
            lines.append(f"V: {self.v_low.get()} - {self.v_high.get()}")
        else:
            lines.append(f"B: {self.h_low.get()} - {self.h_high.get()}")
            lines.append(f"G: {self.s_low.get()} - {self.s_high.get()}")
            lines.append(f"R: {self.v_low.get()} - {self.v_high.get()}")

        output_names = {
            'mask': 'Mask Only',
            'masked': 'Keep In-Range',
            'inverse': 'Keep Out-of-Range'
        }
        lines.append(f"Output: {output_names.get(self.output_mode.get(), self.output_mode.get())}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'use_hsv': self.use_hsv.get(),
            'h_low': self.h_low.get(),
            'h_high': self.h_high.get(),
            's_low': self.s_low.get(),
            's_high': self.s_high.get(),
            'v_low': self.v_low.get(),
            'v_high': self.v_high.get(),
            'output_mode': self.output_mode.get()
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

                    if 'color' in key and 'space' in key:
                        self.use_hsv.set(value.lower() == 'hsv')
                    elif key in ('h', 'b') and '-' in value:
                        low, high = value.split('-')
                        self.h_low.set(max(0, min(255, int(low.strip()))))
                        self.h_high.set(max(0, min(255, int(high.strip()))))
                    elif key in ('s', 'g') and '-' in value:
                        low, high = value.split('-')
                        self.s_low.set(max(0, min(255, int(low.strip()))))
                        self.s_high.set(max(0, min(255, int(high.strip()))))
                    elif key in ('v', 'r') and '-' in value:
                        low, high = value.split('-')
                        self.v_low.set(max(0, min(255, int(low.strip()))))
                        self.v_high.set(max(0, min(255, int(high.strip()))))
                    elif 'output' in key:
                        output_map = {
                            'mask only': 'mask',
                            'keep in-range': 'masked',
                            'keep out-of-range': 'inverse'
                        }
                        self.output_mode.set(output_map.get(value.lower(), 'mask'))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                self._sync_subform_from_vars()
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'use_hsv' in data:
                self.use_hsv.set(bool(data['use_hsv']))
            if 'h_low' in data:
                self.h_low.set(max(0, min(255, int(data['h_low']))))
            if 'h_high' in data:
                self.h_high.set(max(0, min(255, int(data['h_high']))))
            if 's_low' in data:
                self.s_low.set(max(0, min(255, int(data['s_low']))))
            if 's_high' in data:
                self.s_high.set(max(0, min(255, int(data['s_high']))))
            if 'v_low' in data:
                self.v_low.set(max(0, min(255, int(data['v_low']))))
            if 'v_high' in data:
                self.v_high.set(max(0, min(255, int(data['v_high']))))
            if 'output_mode' in data:
                self.output_mode.set(data['output_mode'])

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                self._sync_subform_from_vars()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_vars(self):
        """Sync subform variables from effect variables"""
        if 'use_hsv' in self._subform._vars:
            self._subform._vars['use_hsv'].set(self.use_hsv.get())
        if 'h_low' in self._subform._vars:
            self._subform._vars['h_low'].set(self.h_low.get())
        if 'h_high' in self._subform._vars:
            self._subform._vars['h_high'].set(self.h_high.get())
        if 's_low' in self._subform._vars:
            self._subform._vars['s_low'].set(self.s_low.get())
        if 's_high' in self._subform._vars:
            self._subform._vars['s_high'].set(self.s_high.get())
        if 'v_low' in self._subform._vars:
            self._subform._vars['v_low'].set(self.v_low.get())
        if 'v_high' in self._subform._vars:
            self._subform._vars['v_high'].set(self.v_high.get())
        if 'output_mode' in self._subform._vars:
            output_map = {
                'mask': 'Mask Only',
                'masked': 'Keep In-Range',
                'inverse': 'Keep Out-of-Range'
            }
            self._subform._vars['output_mode'].set(output_map.get(self.output_mode.get(), 'Mask Only'))

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []

        lines.append(f"Color Space: {'HSV' if self.use_hsv.get() else 'BGR'}")

        if self.use_hsv.get():
            lines.append(f"H: {self.h_low.get()} - {self.h_high.get()}")
            lines.append(f"S: {self.s_low.get()} - {self.s_high.get()}")
            lines.append(f"V: {self.v_low.get()} - {self.v_high.get()}")
        else:
            lines.append(f"B: {self.h_low.get()} - {self.h_high.get()}")
            lines.append(f"G: {self.s_low.get()} - {self.s_high.get()}")
            lines.append(f"R: {self.v_low.get()} - {self.v_high.get()}")

        # Map internal values to display names
        output_names = {
            'mask': 'Mask Only',
            'masked': 'Keep In-Range',
            'inverse': 'Keep Out-of-Range'
        }
        lines.append(f"Output: {output_names.get(self.output_mode.get(), self.output_mode.get())}")

        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply inRange color filtering to the frame"""
        if not self.enabled.get():
            return frame

        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Convert to HSV if needed
        if self.use_hsv.get():
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            converted = frame

        # Get range values
        lower = np.array([self.h_low.get(), self.s_low.get(), self.v_low.get()])
        upper = np.array([self.h_high.get(), self.s_high.get(), self.v_high.get()])

        # Create mask
        mask = cv2.inRange(converted, lower, upper)

        # Apply output mode
        output_mode = self.output_mode.get()

        if output_mode == "mask":
            # Return binary mask as BGR
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif output_mode == "masked":
            # Apply mask to original image
            result = cv2.bitwise_and(frame, frame, mask=mask)
        else:  # inverse
            # Invert mask and apply
            inv_mask = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(frame, frame, mask=inv_mask)

        return result

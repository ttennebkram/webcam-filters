"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) effect using OpenCV.

Enhances local contrast using adaptive histogram equalization with clipping.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class CLAHEEffect(BaseUIEffect):
    """Apply CLAHE for adaptive contrast enhancement"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.clip_limit = tk.DoubleVar(value=2.0)
        self.tile_size = tk.IntVar(value=8)
        self.color_mode = tk.StringVar(value="lab")  # lab, hsv, or grayscale

    @classmethod
    def get_name(cls) -> str:
        return "CLAHE: Contrast Enhancement"

    @classmethod
    def get_description(cls) -> str:
        return "CLAHE: Contrast Limited Adaptive Histogram Equalization"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.createCLAHE(clipLimit, tileGridSize)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Clip Limit', 'key': 'clip_limit', 'min': 1.0, 'max': 40.0, 'default': 2.0},
            {'type': 'slider', 'label': 'Tile Size', 'key': 'tile_size', 'min': 2, 'max': 32, 'default': 8},
            {'type': 'dropdown', 'label': 'Apply to', 'key': 'color_mode', 'options': ['lab', 'hsv', 'grayscale'], 'default': 'lab'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'clip_limit': self.clip_limit.get(),
            'tile_size': self.tile_size.get(),
            'color_mode': self.color_mode.get()
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
            if key == 'clip_limit':
                var.trace_add('write', lambda *args: self.clip_limit.set(float(self._subform._vars['clip_limit'].get())))
            elif key == 'tile_size':
                var.trace_add('write', lambda *args: self.tile_size.set(int(self._subform._vars['tile_size'].get())))
            elif key == 'color_mode':
                var.trace_add('write', lambda *args: self.color_mode.set(self._subform._vars['color_mode'].get()))

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
        lines.append(f"Clip Limit: {self.clip_limit.get():.1f}")
        lines.append(f"Tile Size: {self.tile_size.get()}")
        lines.append(f"Apply to: {self.color_mode.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'clip_limit': self.clip_limit.get(),
            'tile_size': self.tile_size.get(),
            'color_mode': self.color_mode.get()
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

                    if 'clip' in key and 'limit' in key:
                        self.clip_limit.set(max(1.0, min(40.0, float(value))))
                    elif 'tile' in key and 'size' in key:
                        self.tile_size.set(max(2, min(32, int(value))))
                    elif 'apply' in key or 'color' in key or 'mode' in key:
                        if value.lower() in ('lab', 'hsv', 'grayscale'):
                            self.color_mode.set(value.lower())

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'clip_limit' in self._subform._vars:
                    self._subform._vars['clip_limit'].set(self.clip_limit.get())
                if 'tile_size' in self._subform._vars:
                    self._subform._vars['tile_size'].set(self.tile_size.get())
                if 'color_mode' in self._subform._vars:
                    self._subform._vars['color_mode'].set(self.color_mode.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'clip_limit' in data:
                self.clip_limit.set(max(1.0, min(40.0, float(data['clip_limit']))))
            if 'tile_size' in data:
                self.tile_size.set(max(2, min(32, int(data['tile_size']))))
            if 'color_mode' in data:
                if data['color_mode'] in ('lab', 'hsv', 'grayscale'):
                    self.color_mode.set(data['color_mode'])

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'clip_limit' in self._subform._vars:
                    self._subform._vars['clip_limit'].set(self.clip_limit.get())
                if 'tile_size' in self._subform._vars:
                    self._subform._vars['tile_size'].set(self.tile_size.get())
                if 'color_mode' in self._subform._vars:
                    self._subform._vars['color_mode'].set(self.color_mode.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Clip Limit: {self.clip_limit.get():.1f}")
        lines.append(f"Tile Size: {self.tile_size.get()}x{self.tile_size.get()}")
        lines.append(f"Apply to: {self.color_mode.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply CLAHE to the frame"""
        if not self.enabled.get():
            return frame

        clip_limit = self.clip_limit.get()
        tile_size = int(self.tile_size.get())
        color_mode = self.color_mode.get()

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )

        if color_mode == "grayscale":
            # Convert to grayscale, apply CLAHE, convert back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = clahe.apply(gray)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        elif color_mode == "lab":
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif color_mode == "hsv":
            # Convert to HSV, apply CLAHE to V channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        else:
            result = frame

        return result

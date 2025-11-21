"""
Morphological erosion effect using OpenCV.

Erodes (shrinks) bright regions in the image.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ErodeEffect(BaseUIEffect):
    """Apply morphological erosion"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Kernel size (must be odd)
        self.kernel_size = tk.IntVar(value=5)

        # Kernel shape
        self.kernel_shape = tk.IntVar(value=0)  # Index into KERNEL_SHAPES

        # Iterations
        self.iterations = tk.IntVar(value=1)

    KERNEL_SHAPES = [
        (cv2.MORPH_RECT, "Rectangle"),
        (cv2.MORPH_ELLIPSE, "Ellipse"),
        (cv2.MORPH_CROSS, "Cross"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "Morph Erode"

    @classmethod
    def get_description(cls) -> str:
        return "Morphological erosion - shrinks bright regions"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.erode(src, kernel, iterations)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Kernel Size', 'key': 'kernel_size', 'min': 1, 'max': 31, 'default': 5},
            {'type': 'dropdown', 'label': 'Kernel Shape', 'key': 'kernel_shape', 'options': ['Rectangle', 'Ellipse', 'Cross'], 'default': 'Rectangle'},
            {'type': 'slider', 'label': 'Iterations', 'key': 'iterations', 'min': 1, 'max': 10, 'default': 1},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1
        shape_idx = self.kernel_shape.get()
        shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Rectangle"
        return {
            'kernel_size': ksize,
            'kernel_shape': shape_name,
            'iterations': self.iterations.get()
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
                def update_ksize(*args):
                    val = int(self._subform._vars['kernel_size'].get())
                    if val % 2 == 0:
                        val += 1
                    self.kernel_size.set(val)
                var.trace_add('write', update_ksize)
            elif key == 'kernel_shape':
                def update_shape(*args):
                    shape_name = self._subform._vars['kernel_shape'].get()
                    for idx, (_, name) in enumerate(self.KERNEL_SHAPES):
                        if name == shape_name:
                            self.kernel_shape.set(idx)
                            break
                var.trace_add('write', update_shape)
            elif key == 'iterations':
                var.trace_add('write', lambda *args: self.iterations.set(int(self._subform._vars['iterations'].get())))

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
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1
        lines.append(f"Kernel Size: {ksize}")
        shape_idx = self.kernel_shape.get()
        shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Rectangle"
        lines.append(f"Kernel Shape: {shape_name}")
        lines.append(f"Iterations: {self.iterations.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1
        shape_idx = self.kernel_shape.get()
        shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Rectangle"
        data = {
            'effect': self.get_name(),
            'kernel_size': ksize,
            'kernel_shape': shape_name,
            'iterations': self.iterations.get()
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

                    if 'kernel size' in key:
                        val = max(1, min(31, int(value)))
                        if val % 2 == 0:
                            val += 1
                        self.kernel_size.set(val)
                    elif 'kernel shape' in key:
                        for idx, (_, name) in enumerate(self.KERNEL_SHAPES):
                            if name.lower() == value.lower():
                                self.kernel_shape.set(idx)
                                break
                    elif 'iterations' in key:
                        self.iterations.set(max(1, min(10, int(value))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size' in self._subform._vars:
                    ksize = self.kernel_size.get()
                    if ksize % 2 == 0:
                        ksize += 1
                    self._subform._vars['kernel_size'].set(ksize)
                if 'kernel_shape' in self._subform._vars:
                    shape_idx = self.kernel_shape.get()
                    shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Rectangle"
                    self._subform._vars['kernel_shape'].set(shape_name)
                if 'iterations' in self._subform._vars:
                    self._subform._vars['iterations'].set(self.iterations.get())
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
                val = max(1, min(31, int(data['kernel_size'])))
                if val % 2 == 0:
                    val += 1
                self.kernel_size.set(val)
            if 'kernel_shape' in data:
                shape_name = data['kernel_shape']
                for idx, (_, name) in enumerate(self.KERNEL_SHAPES):
                    if name.lower() == shape_name.lower():
                        self.kernel_shape.set(idx)
                        break
            if 'iterations' in data:
                self.iterations.set(max(1, min(10, int(data['iterations']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'kernel_size' in self._subform._vars:
                    ksize = self.kernel_size.get()
                    if ksize % 2 == 0:
                        ksize += 1
                    self._subform._vars['kernel_size'].set(ksize)
                if 'kernel_shape' in self._subform._vars:
                    shape_idx = self.kernel_shape.get()
                    shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Rectangle"
                    self._subform._vars['kernel_shape'].set(shape_name)
                if 'iterations' in self._subform._vars:
                    self._subform._vars['iterations'].set(self.iterations.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1
        lines.append(f"Kernel Size: {ksize}")
        shape_idx = self.kernel_shape.get()
        shape_name = self.KERNEL_SHAPES[shape_idx][1] if shape_idx < len(self.KERNEL_SHAPES) else "Unknown"
        lines.append(f"Kernel Shape: {shape_name}")
        lines.append(f"Iterations: {self.iterations.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply erosion to the frame"""
        if not self.enabled.get():
            return frame

        # Get parameters
        ksize = self.kernel_size.get()
        if ksize % 2 == 0:
            ksize += 1

        shape_idx = self.kernel_shape.get()
        shape = self.KERNEL_SHAPES[shape_idx][0]

        iterations = self.iterations.get()

        # Create kernel
        kernel = cv2.getStructuringElement(shape, (ksize, ksize))

        # Apply erosion
        result = cv2.erode(frame, kernel, iterations=iterations)

        return result

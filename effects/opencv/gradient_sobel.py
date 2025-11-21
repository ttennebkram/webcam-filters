"""
Sobel gradient effect using OpenCV.

Computes image gradients using Sobel operator to detect edges.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class GradientSobelEffect(BaseUIEffect):
    """Compute image gradients using Sobel operator"""

    # Depth options for output
    DEPTH_OPTIONS = [
        (cv2.CV_8U, "CV_8U"),
        (cv2.CV_16S, "CV_16S"),
        (cv2.CV_32F, "CV_32F"),
        (cv2.CV_64F, "CV_64F"),
    ]

    # Return mode options
    RETURN_MODES = [
        "gX (uses dx)",
        "gY (uses dy)",
        "Combined (cv2.addWeighted)",
        "Magnitude",
        "Orientation",
        "Mask (angle range filter)",
    ]

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.dx = tk.IntVar(value=1)  # Order of derivative in x
        self.dy = tk.IntVar(value=0)  # Order of derivative in y
        self.ksize = tk.IntVar(value=3)  # Kernel size (1, 3, 5, 7)
        self.depth_index = tk.IntVar(value=3)  # Default to CV_64F
        self.scale = tk.DoubleVar(value=1.0)
        self.delta = tk.DoubleVar(value=0.0)
        self.return_mode_index = tk.IntVar(value=2)  # 0=dx, 1=dy, 2=Combined, 3=Magnitude, 4=Orientation, 5=Mask
        self.weight_x = tk.DoubleVar(value=0.5)
        self.weight_y = tk.DoubleVar(value=0.5)
        self.min_angle = tk.DoubleVar(value=0.0)
        self.max_angle = tk.DoubleVar(value=180.0)

    @classmethod
    def get_name(cls) -> str:
        return "Gradient (Sobel)"

    @classmethod
    def get_description(cls) -> str:
        return "Compute image gradients using Sobel operator for edge detection"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.Sobel(src, ddepth, dx, dy, ksize)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        depth_names = [name for _, name in self.DEPTH_OPTIONS]
        return [
            {'type': 'dropdown', 'label': 'Depth (ddepth)', 'key': 'depth_index', 'options': depth_names, 'default': 'CV_64F'},
            {'type': 'dropdown', 'label': 'dx (x derivative)', 'key': 'dx', 'options': [0, 1, 2], 'default': 1},
            {'type': 'dropdown', 'label': 'dy (y derivative)', 'key': 'dy', 'options': [0, 1, 2], 'default': 0},
            {'type': 'dropdown', 'label': 'Kernel Size', 'key': 'ksize', 'options': [1, 3, 5, 7], 'default': 3},
            {'type': 'slider', 'label': 'Scale', 'key': 'scale', 'min': 0.1, 'max': 10.0, 'default': 1.0, 'step': 0.1},
            {'type': 'slider', 'label': 'Delta', 'key': 'delta', 'min': -128, 'max': 128, 'default': 0.0, 'step': 1.0},
            {'type': 'dropdown', 'label': 'Return Mode', 'key': 'return_mode_index', 'options': self.RETURN_MODES, 'default': 'Combined (cv2.addWeighted)'},
            {'type': 'slider', 'label': 'Weight X', 'key': 'weight_x', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01},
            {'type': 'slider', 'label': 'Weight Y', 'key': 'weight_y', 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.01},
            {'type': 'slider', 'label': 'Min Angle', 'key': 'min_angle', 'min': 0, 'max': 180, 'default': 0.0},
            {'type': 'slider', 'label': 'Max Angle', 'key': 'max_angle', 'min': 0, 'max': 180, 'default': 180.0},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        depth_names = [name for _, name in self.DEPTH_OPTIONS]
        return {
            'depth_index': depth_names[self.depth_index.get()],
            'dx': self.dx.get(),
            'dy': self.dy.get(),
            'ksize': self.ksize.get(),
            'scale': self.scale.get(),
            'delta': self.delta.get(),
            'return_mode_index': self.RETURN_MODES[self.return_mode_index.get()],
            'weight_x': self.weight_x.get(),
            'weight_y': self.weight_y.get(),
            'min_angle': self.min_angle.get(),
            'max_angle': self.max_angle.get()
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
        depth_names = [name for _, name in self.DEPTH_OPTIONS]

        # When subform values change, update effect's tk.Variables
        for key, var in self._subform._vars.items():
            if key == 'depth_index':
                def update_depth(*args):
                    val = self._subform._vars['depth_index'].get()
                    if val in depth_names:
                        self.depth_index.set(depth_names.index(val))
                var.trace_add('write', update_depth)
            elif key == 'dx':
                var.trace_add('write', lambda *args: self.dx.set(int(self._subform._vars['dx'].get())))
            elif key == 'dy':
                var.trace_add('write', lambda *args: self.dy.set(int(self._subform._vars['dy'].get())))
            elif key == 'ksize':
                var.trace_add('write', lambda *args: self.ksize.set(int(self._subform._vars['ksize'].get())))
            elif key == 'scale':
                var.trace_add('write', lambda *args: self.scale.set(float(self._subform._vars['scale'].get())))
            elif key == 'delta':
                var.trace_add('write', lambda *args: self.delta.set(float(self._subform._vars['delta'].get())))
            elif key == 'return_mode_index':
                def update_return_mode(*args):
                    val = self._subform._vars['return_mode_index'].get()
                    if val in self.RETURN_MODES:
                        self.return_mode_index.set(self.RETURN_MODES.index(val))
                var.trace_add('write', update_return_mode)
            elif key == 'weight_x':
                var.trace_add('write', lambda *args: self.weight_x.set(float(self._subform._vars['weight_x'].get())))
            elif key == 'weight_y':
                var.trace_add('write', lambda *args: self.weight_y.set(float(self._subform._vars['weight_y'].get())))
            elif key == 'min_angle':
                var.trace_add('write', lambda *args: self.min_angle.set(float(self._subform._vars['min_angle'].get())))
            elif key == 'max_angle':
                var.trace_add('write', lambda *args: self.max_angle.set(float(self._subform._vars['max_angle'].get())))

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

        depth_name = self.DEPTH_OPTIONS[self.depth_index.get()][1]
        lines.append(f"Depth: {depth_name}")
        lines.append(f"dx: {self.dx.get()}")
        lines.append(f"dy: {self.dy.get()}")
        lines.append(f"Kernel Size: {self.ksize.get()}")
        lines.append(f"Scale: {self.scale.get():.1f}")
        lines.append(f"Delta: {self.delta.get():.1f}")
        lines.append(f"Return Mode: {self.RETURN_MODES[self.return_mode_index.get()]}")
        lines.append(f"Weight X: {self.weight_x.get():.2f}")
        lines.append(f"Weight Y: {self.weight_y.get():.2f}")
        lines.append(f"Min Angle: {self.min_angle.get():.0f}")
        lines.append(f"Max Angle: {self.max_angle.get():.0f}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'depth': self.DEPTH_OPTIONS[self.depth_index.get()][1],
            'dx': self.dx.get(),
            'dy': self.dy.get(),
            'ksize': self.ksize.get(),
            'scale': self.scale.get(),
            'delta': self.delta.get(),
            'return_mode': self.return_mode_index.get(),
            'weight_x': self.weight_x.get(),
            'weight_y': self.weight_y.get(),
            'min_angle': self.min_angle.get(),
            'max_angle': self.max_angle.get()
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
            depth_names = [name for _, name in self.DEPTH_OPTIONS]

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if 'depth' in key and 'ddepth' not in key:
                        if value in depth_names:
                            self.depth_index.set(depth_names.index(value))
                    elif key == 'dx':
                        self.dx.set(max(0, min(2, int(value))))
                    elif key == 'dy':
                        self.dy.set(max(0, min(2, int(value))))
                    elif 'kernel' in key:
                        val = int(value)
                        if val in [1, 3, 5, 7]:
                            self.ksize.set(val)
                    elif 'scale' in key:
                        self.scale.set(max(0.1, min(10.0, float(value))))
                    elif 'delta' in key:
                        self.delta.set(max(-128, min(128, float(value))))
                    elif 'return' in key:
                        for i, mode in enumerate(self.RETURN_MODES):
                            if value in mode or mode in value:
                                self.return_mode_index.set(i)
                                break
                    elif 'weight x' in key:
                        self.weight_x.set(max(0.0, min(1.0, float(value))))
                    elif 'weight y' in key:
                        self.weight_y.set(max(0.0, min(1.0, float(value))))
                    elif 'min' in key and 'angle' in key:
                        self.min_angle.set(max(0, min(180, float(value))))
                    elif 'max' in key and 'angle' in key:
                        self.max_angle.set(max(0, min(180, float(value))))

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
            depth_names = [name for _, name in self.DEPTH_OPTIONS]

            if 'depth' in data:
                if data['depth'] in depth_names:
                    self.depth_index.set(depth_names.index(data['depth']))
            if 'dx' in data:
                self.dx.set(max(0, min(2, int(data['dx']))))
            if 'dy' in data:
                self.dy.set(max(0, min(2, int(data['dy']))))
            if 'ksize' in data:
                val = int(data['ksize'])
                if val in [1, 3, 5, 7]:
                    self.ksize.set(val)
            if 'scale' in data:
                self.scale.set(max(0.1, min(10.0, float(data['scale']))))
            if 'delta' in data:
                self.delta.set(max(-128, min(128, float(data['delta']))))
            if 'return_mode' in data:
                self.return_mode_index.set(max(0, min(5, int(data['return_mode']))))
            if 'weight_x' in data:
                self.weight_x.set(max(0.0, min(1.0, float(data['weight_x']))))
            if 'weight_y' in data:
                self.weight_y.set(max(0.0, min(1.0, float(data['weight_y']))))
            if 'min_angle' in data:
                self.min_angle.set(max(0, min(180, float(data['min_angle']))))
            if 'max_angle' in data:
                self.max_angle.set(max(0, min(180, float(data['max_angle']))))

            # Update subform variables if in edit mode
            self._sync_subform_from_effect()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_effect(self):
        """Sync subform variables from effect variables"""
        if self._current_mode == 'edit' and hasattr(self, '_subform'):
            depth_names = [name for _, name in self.DEPTH_OPTIONS]
            if 'depth_index' in self._subform._vars:
                self._subform._vars['depth_index'].set(depth_names[self.depth_index.get()])
            if 'dx' in self._subform._vars:
                self._subform._vars['dx'].set(str(self.dx.get()))
            if 'dy' in self._subform._vars:
                self._subform._vars['dy'].set(str(self.dy.get()))
            if 'ksize' in self._subform._vars:
                self._subform._vars['ksize'].set(str(self.ksize.get()))
            if 'scale' in self._subform._vars:
                self._subform._vars['scale'].set(self.scale.get())
            if 'delta' in self._subform._vars:
                self._subform._vars['delta'].set(self.delta.get())
            if 'return_mode_index' in self._subform._vars:
                self._subform._vars['return_mode_index'].set(self.RETURN_MODES[self.return_mode_index.get()])
            if 'weight_x' in self._subform._vars:
                self._subform._vars['weight_x'].set(self.weight_x.get())
            if 'weight_y' in self._subform._vars:
                self._subform._vars['weight_y'].set(self.weight_y.get())
            if 'min_angle' in self._subform._vars:
                self._subform._vars['min_angle'].set(self.min_angle.get())
            if 'max_angle' in self._subform._vars:
                self._subform._vars['max_angle'].set(self.max_angle.get())

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []

        # Depth
        depth_idx = self.depth_index.get()
        depth_name = self.DEPTH_OPTIONS[depth_idx][1] if depth_idx < len(self.DEPTH_OPTIONS) else "Unknown"
        lines.append(f"Depth: {depth_name}")

        # dx/dy
        lines.append(f"dx: {self.dx.get()}, dy: {self.dy.get()}")

        # Kernel size
        lines.append(f"Kernel Size: {self.ksize.get()}")

        # Scale and delta
        lines.append(f"Scale: {self.scale.get():.1f}, Delta: {self.delta.get():.1f}")

        # Return mode
        return_modes = ["gX", "gY", "Combined", "Magnitude", "Orientation", "Mask"]
        return_idx = self.return_mode_index.get()
        return_name = return_modes[return_idx] if return_idx < len(return_modes) else "Unknown"
        lines.append(f"Return: {return_name}")

        # Mode-specific params
        if return_idx == 2:  # Combined
            lines.append(f"Weights: X={self.weight_x.get():.2f}, Y={self.weight_y.get():.2f}")
        elif return_idx == 5:  # Mask
            lines.append(f"Angle Range: {self.min_angle.get():.0f} - {self.max_angle.get():.0f}")

        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply Sobel gradient to the frame"""
        # If not enabled, return original frame
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Get parameters
        ddepth, _ = self.DEPTH_OPTIONS[self.depth_index.get()]
        dx = int(self.dx.get())
        dy = int(self.dy.get())
        ksize = self.ksize.get()
        scale = self.scale.get()
        delta = self.delta.get()

        # Get return mode: 0=gX, 1=gY, 2=Combined, 3=Magnitude, 4=Orientation
        return_mode = self.return_mode_index.get()

        if return_mode == 0:
            # gX - use dx value, force dy=0
            if dx == 0:
                dx = 1
            gradient = cv2.Sobel(gray, ddepth=ddepth, dx=dx, dy=0, ksize=ksize, scale=scale, delta=delta)

            # Convert to displayable format
            if ddepth in [cv2.CV_64F, cv2.CV_32F, cv2.CV_16S]:
                result = cv2.convertScaleAbs(gradient)
            else:
                result = gradient

        elif return_mode == 1:
            # gY - use dy value, force dx=0
            if dy == 0:
                dy = 1
            gradient = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=dy, ksize=ksize, scale=scale, delta=delta)

            # Convert to displayable format
            if ddepth in [cv2.CV_64F, cv2.CV_32F, cv2.CV_16S]:
                result = cv2.convertScaleAbs(gradient)
            else:
                result = gradient

        elif return_mode == 2:
            # Combined - weighted average of gX and gY
            gX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Convert to absolute values for combining
            absX = cv2.convertScaleAbs(gX)
            absY = cv2.convertScaleAbs(gY)

            # Combine the sobel X and Y representations into a single image
            weight_x = self.weight_x.get()
            weight_y = self.weight_y.get()
            result = cv2.addWeighted(absX, weight_x, absY, weight_y, 0)

        elif return_mode == 3:
            # Magnitude - sqrt(gX^2 + gY^2)
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute magnitude
            mag = np.sqrt((gX ** 2) + (gY ** 2))

            # Normalize to 0-255
            result = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)

        elif return_mode == 4:
            # Orientation - arctan2(gY, gX) mapped to 0-180
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute orientation in degrees (0-180)
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            # Normalize to 0-255 for display
            result = (orientation * 255 / 180).astype(np.uint8)

        else:  # return_mode == 5
            # Mask - filter by angle range
            gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize, scale=scale, delta=delta)
            gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize, scale=scale, delta=delta)

            # Compute orientation in degrees (0-180)
            orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

            # Get angle range
            lower_angle = self.min_angle.get()
            upper_angle = self.max_angle.get()

            # Find all pixels within the angle boundaries
            idxs = np.where(orientation >= lower_angle, orientation, -1)
            idxs = np.where(orientation <= upper_angle, idxs, -1)
            result = np.zeros(gray.shape, dtype=np.uint8)
            result[idxs > -1] = 255

        # Convert back to BGR (3 identical channels)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

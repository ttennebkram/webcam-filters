"""
Geometric transform effect using OpenCV.

Applies translation, rotation, and scaling transformations.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ScaleAndWarpEffect(BaseUIEffect):
    """Apply geometric transformations: translate, rotate, scale"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Translation
        self.translate_x = tk.IntVar(value=0)
        self.translate_y = tk.IntVar(value=0)

        # Rotation (degrees)
        self.rotation = tk.DoubleVar(value=0.0)

        # Scale
        self.scale = tk.DoubleVar(value=1.0)

        # Center point option
        self.use_image_center = tk.BooleanVar(value=True)
        self.center_x = tk.IntVar(value=width // 2)
        self.center_y = tk.IntVar(value=height // 2)

        # Border mode
        self.border_mode = tk.IntVar(value=0)  # Index into BORDER_MODES

    # Border modes for areas outside the image
    BORDER_MODES = [
        (cv2.BORDER_CONSTANT, "Constant (black)"),
        (cv2.BORDER_REPLICATE, "Replicate edge"),
        (cv2.BORDER_REFLECT, "Reflect"),
        (cv2.BORDER_WRAP, "Wrap"),
    ]

    @classmethod
    def get_name(cls) -> str:
        return "Warp Affine"

    @classmethod
    def get_description(cls) -> str:
        return "Apply translation, rotation, and scaling via affine transform"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.warpAffine(src, M, dsize)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Translate X', 'key': 'translate_x', 'min': -500, 'max': 500, 'default': 0},
            {'type': 'slider', 'label': 'Translate Y', 'key': 'translate_y', 'min': -500, 'max': 500, 'default': 0},
            {'type': 'slider', 'label': 'Rotation', 'key': 'rotation', 'min': -180, 'max': 180, 'default': 0},
            {'type': 'slider', 'label': 'Scale', 'key': 'scale', 'min': 0.1, 'max': 4.0, 'default': 1.0},
            {'type': 'checkbox', 'label': 'Use Image Center', 'key': 'use_image_center', 'default': True},
            {'type': 'dropdown', 'label': 'Border Mode', 'key': 'border_mode', 'options': [name for _, name in cls.BORDER_MODES], 'default': 'Constant (black)'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        border_idx = self.border_mode.get()

        return {
            'translate_x': self.translate_x.get(),
            'translate_y': self.translate_y.get(),
            'rotation': self.rotation.get(),
            'scale': self.scale.get(),
            'use_image_center': self.use_image_center.get(),
            'border_mode': self.BORDER_MODES[border_idx][1] if border_idx < len(self.BORDER_MODES) else 'Constant (black)',
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
            if key == 'translate_x':
                var.trace_add('write', lambda *args: self.translate_x.set(int(self._subform._vars['translate_x'].get())))
            elif key == 'translate_y':
                var.trace_add('write', lambda *args: self.translate_y.set(int(self._subform._vars['translate_y'].get())))
            elif key == 'rotation':
                var.trace_add('write', lambda *args: self.rotation.set(float(self._subform._vars['rotation'].get())))
            elif key == 'scale':
                var.trace_add('write', lambda *args: self.scale.set(float(self._subform._vars['scale'].get())))
            elif key == 'use_image_center':
                var.trace_add('write', lambda *args: self.use_image_center.set(self._subform._vars['use_image_center'].get()))
            elif key == 'border_mode':
                var.trace_add('write', lambda *args: self._update_border_mode())

    def _update_border_mode(self):
        """Update border mode from subform value"""
        value = self._subform._vars['border_mode'].get()
        for idx, (_, name) in enumerate(self.BORDER_MODES):
            if name == value:
                self.border_mode.set(idx)
                break

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
        lines.append(f"Translate X: {self.translate_x.get()}")
        lines.append(f"Translate Y: {self.translate_y.get()}")
        lines.append(f"Rotation: {self.rotation.get():.1f}")
        lines.append(f"Scale: {self.scale.get():.2f}")
        lines.append(f"Use Image Center: {'Yes' if self.use_image_center.get() else 'No'}")

        border_idx = self.border_mode.get()
        lines.append(f"Border Mode: {self.BORDER_MODES[border_idx][1] if border_idx < len(self.BORDER_MODES) else 'Unknown'}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        border_idx = self.border_mode.get()

        data = {
            'effect': self.get_name(),
            'translate_x': self.translate_x.get(),
            'translate_y': self.translate_y.get(),
            'rotation': self.rotation.get(),
            'scale': self.scale.get(),
            'use_image_center': self.use_image_center.get(),
            'border_mode': self.BORDER_MODES[border_idx][1] if border_idx < len(self.BORDER_MODES) else 'Constant (black)',
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

                    if 'translate' in key and 'x' in key:
                        self.translate_x.set(max(-500, min(500, int(value))))
                    elif 'translate' in key and 'y' in key:
                        self.translate_y.set(max(-500, min(500, int(value))))
                    elif 'rotation' in key:
                        self.rotation.set(max(-180, min(180, float(value))))
                    elif 'scale' in key:
                        self.scale.set(max(0.1, min(4.0, float(value))))
                    elif 'use' in key and 'center' in key:
                        self.use_image_center.set(value.lower() in ('yes', 'true', '1'))
                    elif 'border' in key and 'mode' in key:
                        for idx, (_, name) in enumerate(self.BORDER_MODES):
                            if name == value:
                                self.border_mode.set(idx)
                                break

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

            if 'translate_x' in data:
                self.translate_x.set(max(-500, min(500, int(data['translate_x']))))
            if 'translate_y' in data:
                self.translate_y.set(max(-500, min(500, int(data['translate_y']))))
            if 'rotation' in data:
                self.rotation.set(max(-180, min(180, float(data['rotation']))))
            if 'scale' in data:
                self.scale.set(max(0.1, min(4.0, float(data['scale']))))
            if 'use_image_center' in data:
                self.use_image_center.set(bool(data['use_image_center']))
            if 'border_mode' in data:
                for idx, (_, name) in enumerate(self.BORDER_MODES):
                    if name == data['border_mode']:
                        self.border_mode.set(idx)
                        break

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                self._sync_subform_from_vars()
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def _sync_subform_from_vars(self):
        """Sync subform variables from effect variables"""
        if 'translate_x' in self._subform._vars:
            self._subform._vars['translate_x'].set(self.translate_x.get())
        if 'translate_y' in self._subform._vars:
            self._subform._vars['translate_y'].set(self.translate_y.get())
        if 'rotation' in self._subform._vars:
            self._subform._vars['rotation'].set(self.rotation.get())
        if 'scale' in self._subform._vars:
            self._subform._vars['scale'].set(self.scale.get())
        if 'use_image_center' in self._subform._vars:
            self._subform._vars['use_image_center'].set(self.use_image_center.get())
        if 'border_mode' in self._subform._vars:
            idx = self.border_mode.get()
            self._subform._vars['border_mode'].set(self.BORDER_MODES[idx][1] if idx < len(self.BORDER_MODES) else 'Constant (black)')

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []

        # Translation
        tx = self.translate_x.get()
        ty = self.translate_y.get()
        lines.append(f"Translation: ({tx}, {ty})")

        # Rotation
        rotation = self.rotation.get()
        lines.append(f"Rotation: {rotation:.1f}")

        # Scale
        scale = self.scale.get()
        lines.append(f"Scale: {scale:.2f}x")

        # Center
        if self.use_image_center.get():
            lines.append("Center: Image Center")
        else:
            cx = self.center_x.get()
            cy = self.center_y.get()
            lines.append(f"Center: ({cx}, {cy})")

        # Border mode - translate index to display name
        border_idx = self.border_mode.get()
        border_name = self.BORDER_MODES[border_idx][1] if border_idx < len(self.BORDER_MODES) else "Unknown"
        lines.append(f"Border Mode: {border_name}")

        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply geometric transformation to the frame"""
        if not self.enabled.get():
            return frame

        height, width = frame.shape[:2]

        # Get parameters
        tx = self.translate_x.get()
        ty = self.translate_y.get()
        angle = self.rotation.get()
        scale = self.scale.get()

        # Determine center point
        if self.use_image_center.get():
            cx, cy = width / 2, height / 2
        else:
            try:
                cx = int(self.center_x.get())
                cy = int(self.center_y.get())
            except (ValueError, tk.TclError):
                cx, cy = width / 2, height / 2

        # Build transformation matrix
        # getRotationMatrix2D handles rotation and scale around center
        # Negate angle so positive = clockwise (more intuitive)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, scale)

        # Add translation
        M[0, 2] += tx
        M[1, 2] += ty

        # Get border mode
        border_idx = self.border_mode.get()
        border_mode = self.BORDER_MODES[border_idx][0]

        # Apply transformation
        result = cv2.warpAffine(
            frame,
            M,
            (width, height),
            borderMode=border_mode
        )

        return result

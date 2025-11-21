"""
ORB (Oriented FAST and Rotated BRIEF) feature detection using OpenCV.

Fast, free alternative to SIFT/SURF. Great for real-time applications.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ORBEffect(BaseUIEffect):
    """Detect and visualize ORB keypoints"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.n_features = tk.IntVar(value=500)
        self.scale_factor = tk.DoubleVar(value=1.2)
        self.n_levels = tk.IntVar(value=8)
        self.edge_threshold = tk.IntVar(value=31)
        self.first_level = tk.IntVar(value=0)
        self.wta_k = tk.IntVar(value=2)
        self.patch_size = tk.IntVar(value=31)
        self.fast_threshold = tk.IntVar(value=20)
        self.show_rich_keypoints = tk.BooleanVar(value=True)
        self.keypoint_color = tk.StringVar(value="green")

    @classmethod
    def get_name(cls) -> str:
        return "ORB: Fast Real-time Features"

    @classmethod
    def get_description(cls) -> str:
        return "Fast, free keypoint detection for real-time use"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.ORB_create(nfeatures, scaleFactor, nlevels, ...)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Max Features', 'key': 'n_features', 'min': 10, 'max': 5000, 'default': 500},
            {'type': 'slider', 'label': 'Scale Factor', 'key': 'scale_factor', 'min': 1.01, 'max': 2.0, 'default': 1.2, 'resolution': 0.01},
            {'type': 'slider', 'label': 'Pyramid Levels', 'key': 'n_levels', 'min': 1, 'max': 16, 'default': 8},
            {'type': 'slider', 'label': 'FAST Threshold', 'key': 'fast_threshold', 'min': 1, 'max': 100, 'default': 20},
            {'type': 'checkbox', 'label': 'Show Size & Orientation', 'key': 'show_rich_keypoints', 'default': True},
            {'type': 'dropdown', 'label': 'Keypoint Color', 'key': 'keypoint_color', 'options': ['green', 'red', 'blue', 'yellow', 'white'], 'default': 'green'},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'n_features': self.n_features.get(),
            'scale_factor': self.scale_factor.get(),
            'n_levels': self.n_levels.get(),
            'fast_threshold': self.fast_threshold.get(),
            'show_rich_keypoints': self.show_rich_keypoints.get(),
            'keypoint_color': self.keypoint_color.get()
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
            if key == 'n_features':
                var.trace_add('write', lambda *args: self.n_features.set(int(self._subform._vars['n_features'].get())))
            elif key == 'scale_factor':
                var.trace_add('write', lambda *args: self.scale_factor.set(float(self._subform._vars['scale_factor'].get())))
            elif key == 'n_levels':
                var.trace_add('write', lambda *args: self.n_levels.set(int(self._subform._vars['n_levels'].get())))
            elif key == 'fast_threshold':
                var.trace_add('write', lambda *args: self.fast_threshold.set(int(self._subform._vars['fast_threshold'].get())))
            elif key == 'show_rich_keypoints':
                var.trace_add('write', lambda *args: self.show_rich_keypoints.set(self._subform._vars['show_rich_keypoints'].get()))
            elif key == 'keypoint_color':
                var.trace_add('write', lambda *args: self.keypoint_color.set(self._subform._vars['keypoint_color'].get()))

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
        lines.append(f"Max Features: {self.n_features.get()}")
        lines.append(f"Scale Factor: {self.scale_factor.get():.2f}")
        lines.append(f"Pyramid Levels: {self.n_levels.get()}")
        lines.append(f"FAST Threshold: {self.fast_threshold.get()}")
        lines.append(f"Show Size & Orientation: {'Yes' if self.show_rich_keypoints.get() else 'No'}")
        lines.append(f"Keypoint Color: {self.keypoint_color.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'n_features': self.n_features.get(),
            'scale_factor': self.scale_factor.get(),
            'n_levels': self.n_levels.get(),
            'fast_threshold': self.fast_threshold.get(),
            'show_rich_keypoints': self.show_rich_keypoints.get(),
            'keypoint_color': self.keypoint_color.get()
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

                    if 'max features' in key or 'features' in key:
                        self.n_features.set(max(10, min(5000, int(value))))
                    elif 'scale factor' in key:
                        self.scale_factor.set(max(1.01, min(2.0, float(value))))
                    elif 'pyramid' in key or 'levels' in key:
                        self.n_levels.set(max(1, min(16, int(value))))
                    elif 'fast' in key:
                        self.fast_threshold.set(max(1, min(100, int(value))))
                    elif 'size' in key or 'orientation' in key:
                        self.show_rich_keypoints.set(value.lower() in ('yes', 'true', '1'))
                    elif 'color' in key:
                        if value.lower() in ['green', 'red', 'blue', 'yellow', 'white']:
                            self.keypoint_color.set(value.lower())

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'n_features' in self._subform._vars:
                    self._subform._vars['n_features'].set(self.n_features.get())
                if 'scale_factor' in self._subform._vars:
                    self._subform._vars['scale_factor'].set(self.scale_factor.get())
                if 'n_levels' in self._subform._vars:
                    self._subform._vars['n_levels'].set(self.n_levels.get())
                if 'fast_threshold' in self._subform._vars:
                    self._subform._vars['fast_threshold'].set(self.fast_threshold.get())
                if 'show_rich_keypoints' in self._subform._vars:
                    self._subform._vars['show_rich_keypoints'].set(self.show_rich_keypoints.get())
                if 'keypoint_color' in self._subform._vars:
                    self._subform._vars['keypoint_color'].set(self.keypoint_color.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'n_features' in data:
                self.n_features.set(max(10, min(5000, int(data['n_features']))))
            if 'scale_factor' in data:
                self.scale_factor.set(max(1.01, min(2.0, float(data['scale_factor']))))
            if 'n_levels' in data:
                self.n_levels.set(max(1, min(16, int(data['n_levels']))))
            if 'fast_threshold' in data:
                self.fast_threshold.set(max(1, min(100, int(data['fast_threshold']))))
            if 'show_rich_keypoints' in data:
                self.show_rich_keypoints.set(bool(data['show_rich_keypoints']))
            if 'keypoint_color' in data:
                if data['keypoint_color'] in ['green', 'red', 'blue', 'yellow', 'white']:
                    self.keypoint_color.set(data['keypoint_color'])

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'n_features' in self._subform._vars:
                    self._subform._vars['n_features'].set(self.n_features.get())
                if 'scale_factor' in self._subform._vars:
                    self._subform._vars['scale_factor'].set(self.scale_factor.get())
                if 'n_levels' in self._subform._vars:
                    self._subform._vars['n_levels'].set(self.n_levels.get())
                if 'fast_threshold' in self._subform._vars:
                    self._subform._vars['fast_threshold'].set(self.fast_threshold.get())
                if 'show_rich_keypoints' in self._subform._vars:
                    self._subform._vars['show_rich_keypoints'].set(self.show_rich_keypoints.get())
                if 'keypoint_color' in self._subform._vars:
                    self._subform._vars['keypoint_color'].set(self.keypoint_color.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Max Features: {self.n_features.get()}")
        lines.append(f"Scale Factor: {self.scale_factor.get():.2f}")
        lines.append(f"Pyramid Levels: {self.n_levels.get()}")
        lines.append(f"FAST Threshold: {self.fast_threshold.get()}")
        lines.append(f"Show Size & Orientation: {'Yes' if self.show_rich_keypoints.get() else 'No'}")
        lines.append(f"Keypoint Color: {self.keypoint_color.get()}")
        return '\n'.join(lines)

    def _get_color_bgr(self):
        """Get BGR color tuple from color name"""
        colors = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255)
        }
        return colors.get(self.keypoint_color.get(), (0, 255, 0))

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw ORB keypoints"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create ORB detector
        orb = cv2.ORB_create(
            nfeatures=self.n_features.get(),
            scaleFactor=self.scale_factor.get(),
            nlevels=self.n_levels.get(),
            edgeThreshold=self.edge_threshold.get(),
            firstLevel=self.first_level.get(),
            WTA_K=self.wta_k.get(),
            patchSize=self.patch_size.get(),
            fastThreshold=self.fast_threshold.get()
        )

        # Detect keypoints
        keypoints = orb.detect(gray, None)

        # Draw keypoints
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if self.show_rich_keypoints.get() else 0
        result = cv2.drawKeypoints(
            frame,
            keypoints,
            None,
            color=self._get_color_bgr(),
            flags=flags
        )

        return result

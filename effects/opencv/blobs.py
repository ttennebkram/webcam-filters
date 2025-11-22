"""
Blob detection effect using OpenCV SimpleBlobDetector.

Detects blobs (connected regions) in images based on various properties.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class BlobsEffect(BaseUIEffect):
    """Detect and draw blobs using SimpleBlobDetector"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Input: background option
        self.show_original = tk.BooleanVar(value=True)

        # Output: draw or store raw
        self.draw_features = tk.BooleanVar(value=True)

        # Storage for detected features (for pipeline use)
        self.detected_keypoints = None

        # Threshold parameters
        self.min_threshold = tk.IntVar(value=10)
        self.max_threshold = tk.IntVar(value=200)
        self.threshold_step = tk.IntVar(value=10)

        # Filter by Area
        self.filter_by_area = tk.BooleanVar(value=True)
        self.min_area = tk.IntVar(value=100)
        self.max_area = tk.IntVar(value=5000)

        # Filter by Circularity
        self.filter_by_circularity = tk.BooleanVar(value=False)
        self.min_circularity = tk.DoubleVar(value=0.1)

        # Filter by Convexity
        self.filter_by_convexity = tk.BooleanVar(value=False)
        self.min_convexity = tk.DoubleVar(value=0.87)

        # Filter by Inertia
        self.filter_by_inertia = tk.BooleanVar(value=False)
        self.min_inertia_ratio = tk.DoubleVar(value=0.01)

        # Filter by Color
        self.filter_by_color = tk.BooleanVar(value=False)
        self.blob_color = tk.IntVar(value=0)  # 0 = dark, 255 = light

        # Drawing options
        self.circle_color_b = tk.IntVar(value=0)
        self.circle_color_g = tk.IntVar(value=0)
        self.circle_color_r = tk.IntVar(value=255)

    @classmethod
    def get_name(cls) -> str:
        return "Blob Detector"

    @classmethod
    def get_description(cls) -> str:
        return "Detect blobs using SimpleBlobDetector with configurable filters"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.SimpleBlobDetector_create(params) / detector.detect(image)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'checkbox', 'label': 'Keep Original', 'key': 'show_original', 'default': True},
            {'type': 'checkbox', 'label': 'Draw Features', 'key': 'draw_features', 'default': True},
            {'type': 'slider', 'label': 'Min Threshold', 'key': 'min_threshold', 'min': 0, 'max': 255, 'default': 10},
            {'type': 'slider', 'label': 'Max Threshold', 'key': 'max_threshold', 'min': 0, 'max': 255, 'default': 200},
            {'type': 'checkbox', 'label': 'Filter by Area', 'key': 'filter_by_area', 'default': True},
            {'type': 'slider', 'label': 'Min Area', 'key': 'min_area', 'min': 1, 'max': 10000, 'default': 100},
            {'type': 'slider', 'label': 'Max Area', 'key': 'max_area', 'min': 1, 'max': 50000, 'default': 5000},
            {'type': 'checkbox', 'label': 'Filter by Circularity', 'key': 'filter_by_circularity', 'default': False},
            {'type': 'slider', 'label': 'Min Circularity', 'key': 'min_circularity', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'resolution': 0.01},
            {'type': 'checkbox', 'label': 'Filter by Convexity', 'key': 'filter_by_convexity', 'default': False},
            {'type': 'slider', 'label': 'Min Convexity', 'key': 'min_convexity', 'min': 0.0, 'max': 1.0, 'default': 0.87, 'resolution': 0.01},
            {'type': 'checkbox', 'label': 'Filter by Inertia', 'key': 'filter_by_inertia', 'default': False},
            {'type': 'slider', 'label': 'Min Inertia Ratio', 'key': 'min_inertia_ratio', 'min': 0.0, 'max': 1.0, 'default': 0.01, 'resolution': 0.01},
            {'type': 'slider', 'label': 'Color B', 'key': 'circle_color_b', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color G', 'key': 'circle_color_g', 'min': 0, 'max': 255, 'default': 0},
            {'type': 'slider', 'label': 'Color R', 'key': 'circle_color_r', 'min': 0, 'max': 255, 'default': 255},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'show_original': self.show_original.get(),
            'draw_features': self.draw_features.get(),
            'min_threshold': self.min_threshold.get(),
            'max_threshold': self.max_threshold.get(),
            'filter_by_area': self.filter_by_area.get(),
            'min_area': self.min_area.get(),
            'max_area': self.max_area.get(),
            'filter_by_circularity': self.filter_by_circularity.get(),
            'min_circularity': self.min_circularity.get(),
            'filter_by_convexity': self.filter_by_convexity.get(),
            'min_convexity': self.min_convexity.get(),
            'filter_by_inertia': self.filter_by_inertia.get(),
            'min_inertia_ratio': self.min_inertia_ratio.get(),
            'circle_color_b': self.circle_color_b.get(),
            'circle_color_g': self.circle_color_g.get(),
            'circle_color_r': self.circle_color_r.get()
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
        for key, var in self._subform._vars.items():
            if key == 'show_original':
                var.trace_add('write', lambda *args: self.show_original.set(self._subform._vars['show_original'].get()))
            elif key == 'draw_features':
                var.trace_add('write', lambda *args: self.draw_features.set(self._subform._vars['draw_features'].get()))
            elif key == 'min_threshold':
                var.trace_add('write', lambda *args: self.min_threshold.set(int(self._subform._vars['min_threshold'].get())))
            elif key == 'max_threshold':
                var.trace_add('write', lambda *args: self.max_threshold.set(int(self._subform._vars['max_threshold'].get())))
            elif key == 'filter_by_area':
                var.trace_add('write', lambda *args: self.filter_by_area.set(self._subform._vars['filter_by_area'].get()))
            elif key == 'min_area':
                var.trace_add('write', lambda *args: self.min_area.set(int(self._subform._vars['min_area'].get())))
            elif key == 'max_area':
                var.trace_add('write', lambda *args: self.max_area.set(int(self._subform._vars['max_area'].get())))
            elif key == 'filter_by_circularity':
                var.trace_add('write', lambda *args: self.filter_by_circularity.set(self._subform._vars['filter_by_circularity'].get()))
            elif key == 'min_circularity':
                var.trace_add('write', lambda *args: self.min_circularity.set(self._subform._vars['min_circularity'].get()))
            elif key == 'filter_by_convexity':
                var.trace_add('write', lambda *args: self.filter_by_convexity.set(self._subform._vars['filter_by_convexity'].get()))
            elif key == 'min_convexity':
                var.trace_add('write', lambda *args: self.min_convexity.set(self._subform._vars['min_convexity'].get()))
            elif key == 'filter_by_inertia':
                var.trace_add('write', lambda *args: self.filter_by_inertia.set(self._subform._vars['filter_by_inertia'].get()))
            elif key == 'min_inertia_ratio':
                var.trace_add('write', lambda *args: self.min_inertia_ratio.set(self._subform._vars['min_inertia_ratio'].get()))
            elif key == 'circle_color_b':
                var.trace_add('write', lambda *args: self.circle_color_b.set(int(self._subform._vars['circle_color_b'].get())))
            elif key == 'circle_color_g':
                var.trace_add('write', lambda *args: self.circle_color_g.set(int(self._subform._vars['circle_color_g'].get())))
            elif key == 'circle_color_r':
                var.trace_add('write', lambda *args: self.circle_color_r.set(int(self._subform._vars['circle_color_r'].get())))

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

        self._update_vars_from_subform()

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())
        lines.append(f"Keep Original: {'Yes' if self.show_original.get() else 'No'}")
        lines.append(f"Output: {'Draw Features' if self.draw_features.get() else 'Raw Values Only'}")
        lines.append(f"Threshold: {self.min_threshold.get()} - {self.max_threshold.get()}")
        if self.filter_by_area.get():
            lines.append(f"Area: {self.min_area.get()} - {self.max_area.get()}")
        if self.filter_by_circularity.get():
            lines.append(f"Min Circularity: {self.min_circularity.get():.2f}")
        if self.filter_by_convexity.get():
            lines.append(f"Min Convexity: {self.min_convexity.get():.2f}")
        if self.filter_by_inertia.get():
            lines.append(f"Min Inertia Ratio: {self.min_inertia_ratio.get():.2f}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            **self.get_current_data()
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
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('Threshold:'):
                    parts = line.split(':')[1].strip().split('-')
                    if len(parts) == 2:
                        self.min_threshold.set(max(0, min(255, int(parts[0].strip()))))
                        self.max_threshold.set(max(0, min(255, int(parts[1].strip()))))
                elif line.startswith('Area:'):
                    parts = line.split(':')[1].strip().split('-')
                    if len(parts) == 2:
                        self.min_area.set(max(1, min(10000, int(parts[0].strip()))))
                        self.max_area.set(max(1, min(50000, int(parts[1].strip()))))
                elif line.startswith('Min Circularity:'):
                    value = float(line.split(':')[1].strip())
                    self.min_circularity.set(max(0.0, min(1.0, value)))
                elif line.startswith('Min Convexity:'):
                    value = float(line.split(':')[1].strip())
                    self.min_convexity.set(max(0.0, min(1.0, value)))
                elif line.startswith('Min Inertia Ratio:'):
                    value = float(line.split(':')[1].strip())
                    self.min_inertia_ratio.set(max(0.0, min(1.0, value)))
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'show_original' in data:
                self.show_original.set(bool(data['show_original']))
            if 'draw_features' in data:
                self.draw_features.set(bool(data['draw_features']))
            if 'min_threshold' in data:
                self.min_threshold.set(max(0, min(255, int(data['min_threshold']))))
            if 'max_threshold' in data:
                self.max_threshold.set(max(0, min(255, int(data['max_threshold']))))
            if 'filter_by_area' in data:
                self.filter_by_area.set(bool(data['filter_by_area']))
            if 'min_area' in data:
                self.min_area.set(max(1, min(10000, int(data['min_area']))))
            if 'max_area' in data:
                self.max_area.set(max(1, min(50000, int(data['max_area']))))
            if 'filter_by_circularity' in data:
                self.filter_by_circularity.set(bool(data['filter_by_circularity']))
            if 'min_circularity' in data:
                self.min_circularity.set(max(0.0, min(1.0, float(data['min_circularity']))))
            if 'filter_by_convexity' in data:
                self.filter_by_convexity.set(bool(data['filter_by_convexity']))
            if 'min_convexity' in data:
                self.min_convexity.set(max(0.0, min(1.0, float(data['min_convexity']))))
            if 'filter_by_inertia' in data:
                self.filter_by_inertia.set(bool(data['filter_by_inertia']))
            if 'min_inertia_ratio' in data:
                self.min_inertia_ratio.set(max(0.0, min(1.0, float(data['min_inertia_ratio']))))
            if 'circle_color_b' in data:
                self.circle_color_b.set(max(0, min(255, int(data['circle_color_b']))))
            if 'circle_color_g' in data:
                self.circle_color_g.set(max(0, min(255, int(data['circle_color_g']))))
            if 'circle_color_r' in data:
                self.circle_color_r.set(max(0, min(255, int(data['circle_color_r']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                for key in self.get_current_data().keys():
                    if key in self._subform._vars:
                        self._subform._vars[key].set(getattr(self, key).get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Keep Original: {'Yes' if self.show_original.get() else 'No'}")
        lines.append(f"Output: {'Draw Features' if self.draw_features.get() else 'Raw Values Only'}")
        lines.append(f"Threshold: {self.min_threshold.get()} - {self.max_threshold.get()}")
        if self.filter_by_area.get():
            lines.append(f"Area: {self.min_area.get()} - {self.max_area.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Detect and draw blobs on the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Create output image
        if self.show_original.get():
            result = frame.copy()
        else:
            result = np.zeros_like(frame)

        # Set up SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Threshold parameters
        params.minThreshold = self.min_threshold.get()
        params.maxThreshold = self.max_threshold.get()
        params.thresholdStep = self.threshold_step.get()

        # Filter by Area
        params.filterByArea = self.filter_by_area.get()
        params.minArea = self.min_area.get()
        params.maxArea = self.max_area.get()

        # Filter by Circularity
        params.filterByCircularity = self.filter_by_circularity.get()
        params.minCircularity = self.min_circularity.get()

        # Filter by Convexity
        params.filterByConvexity = self.filter_by_convexity.get()
        params.minConvexity = self.min_convexity.get()

        # Filter by Inertia
        params.filterByInertia = self.filter_by_inertia.get()
        params.minInertiaRatio = self.min_inertia_ratio.get()

        # Filter by Color
        params.filterByColor = self.filter_by_color.get()
        params.blobColor = self.blob_color.get()

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(gray)

        # Store for pipeline use
        self.detected_keypoints = keypoints

        # Draw blobs
        if self.draw_features.get() and keypoints:
            # Get color
            try:
                color_b = self.circle_color_b.get()
            except Exception:
                color_b = 0
            try:
                color_g = self.circle_color_g.get()
            except Exception:
                color_g = 0
            try:
                color_r = self.circle_color_r.get()
            except Exception:
                color_r = 255
            color = (color_b, color_g, color_r)

            # Draw keypoints as circles
            result = cv2.drawKeypoints(
                result, keypoints, np.array([]), color,
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

        return result

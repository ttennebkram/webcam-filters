"""
Connected components labeling effect using OpenCV.

Labels connected regions (blobs) in a binary image with different colors.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json


class ConnectedComponentsEffect(BaseUIEffect):
    """Label connected components with different colors"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Threshold for binarization
        self.threshold = tk.IntVar(value=127)

        # Invert threshold
        self.invert = tk.BooleanVar(value=False)

        # Connectivity (4 or 8)
        self.connectivity = tk.IntVar(value=8)

        # Min area filter
        self.min_area = tk.IntVar(value=0)

        # Storage for labels
        self.num_labels = 0
        self.labels = None
        self.stats = None
        self.centroids = None

    @classmethod
    def get_name(cls) -> str:
        return "Connected Components"

    @classmethod
    def get_description(cls) -> str:
        return "Label connected regions with different colors"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.connectedComponentsWithStats(image)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Threshold', 'key': 'threshold', 'min': 0, 'max': 255, 'default': 127},
            {'type': 'checkbox', 'label': 'Invert Threshold', 'key': 'invert', 'default': False},
            {'type': 'dropdown', 'label': 'Connectivity', 'key': 'connectivity', 'options': [4, 8], 'default': 8},
            {'type': 'slider', 'label': 'Min Area', 'key': 'min_area', 'min': 0, 'max': 1000, 'default': 0},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'threshold': self.threshold.get(),
            'invert': self.invert.get(),
            'connectivity': self.connectivity.get(),
            'min_area': self.min_area.get()
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
            if key == 'threshold':
                var.trace_add('write', lambda *args: self.threshold.set(int(self._subform._vars['threshold'].get())))
            elif key == 'invert':
                var.trace_add('write', lambda *args: self.invert.set(self._subform._vars['invert'].get()))
            elif key == 'connectivity':
                var.trace_add('write', lambda *args: self.connectivity.set(int(self._subform._vars['connectivity'].get())))
            elif key == 'min_area':
                var.trace_add('write', lambda *args: self.min_area.set(int(self._subform._vars['min_area'].get())))

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
        lines.append(f"Threshold: {self.threshold.get()}")
        lines.append(f"Invert Threshold: {'Yes' if self.invert.get() else 'No'}")
        lines.append(f"Connectivity: {self.connectivity.get()}")
        lines.append(f"Min Area: {self.min_area.get()}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'threshold': self.threshold.get(),
            'invert': self.invert.get(),
            'connectivity': self.connectivity.get(),
            'min_area': self.min_area.get()
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

                    if 'threshold' in key and 'invert' not in key:
                        self.threshold.set(max(0, min(255, int(value))))
                    elif 'invert' in key:
                        self.invert.set(value.lower() in ('yes', 'true', '1'))
                    elif 'connectivity' in key:
                        val = int(value)
                        if val <= 4:
                            val = 4
                        else:
                            val = 8
                        self.connectivity.set(val)
                    elif 'min area' in key or 'area' in key:
                        self.min_area.set(max(0, min(1000, int(value))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'threshold' in self._subform._vars:
                    self._subform._vars['threshold'].set(self.threshold.get())
                if 'invert' in self._subform._vars:
                    self._subform._vars['invert'].set(self.invert.get())
                if 'connectivity' in self._subform._vars:
                    self._subform._vars['connectivity'].set(str(self.connectivity.get()))
                if 'min_area' in self._subform._vars:
                    self._subform._vars['min_area'].set(self.min_area.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'threshold' in data:
                self.threshold.set(max(0, min(255, int(data['threshold']))))
            if 'invert' in data:
                self.invert.set(bool(data['invert']))
            if 'connectivity' in data:
                val = int(data['connectivity'])
                if val <= 4:
                    val = 4
                else:
                    val = 8
                self.connectivity.set(val)
            if 'min_area' in data:
                self.min_area.set(max(0, min(1000, int(data['min_area']))))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'threshold' in self._subform._vars:
                    self._subform._vars['threshold'].set(self.threshold.get())
                if 'invert' in self._subform._vars:
                    self._subform._vars['invert'].set(self.invert.get())
                if 'connectivity' in self._subform._vars:
                    self._subform._vars['connectivity'].set(str(self.connectivity.get()))
                if 'min_area' in self._subform._vars:
                    self._subform._vars['min_area'].set(self.min_area.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Threshold: {self.threshold.get()}")
        lines.append(f"Invert Threshold: {'Yes' if self.invert.get() else 'No'}")
        lines.append(f"Connectivity: {self.connectivity.get()}")
        lines.append(f"Min Area: {self.min_area.get()}")
        return '\n'.join(lines)

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Label connected components and colorize them"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply threshold
        threshold = self.threshold.get()
        if self.invert.get():
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Get connected components with stats
        connectivity = self.connectivity.get()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity, cv2.CV_32S
        )

        # Store for pipeline use
        self.num_labels = num_labels
        self.labels = labels
        self.stats = stats
        self.centroids = centroids

        # Create colored output
        # Generate random colors for each label (excluding background)
        np.random.seed(42)  # Consistent colors
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black

        # Filter by min area
        min_area = self.min_area.get()
        if min_area > 0:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    colors[i] = [0, 0, 0]  # Set small components to black

        # Map labels to colors
        result = colors[labels]

        return result

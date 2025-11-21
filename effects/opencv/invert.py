"""
Invert effect - inverts all pixel values in the frame.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class InvertEffect(BaseUIEffect):
    """Effect that inverts all pixel values (negative image)"""

    def __init__(self, width: int, height: int, root=None, **kwargs):
        """Initialize the invert effect

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            root: Tkinter root window
        """
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

    @classmethod
    def get_name(cls) -> str:
        return "Invert"

    @classmethod
    def get_description(cls) -> str:
        return "Inverts all pixel values to create a negative image"

    @classmethod
    def get_method_signature(cls) -> str:
        return "255 - pixel value; works on grayscale or color images"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        # No parameters - just uses enabled checkbox
        return []

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {}

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

        return self.control_panel

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

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name()
        }

        text = json.dumps(data, indent=2)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _paste_text(self):
        """Paste settings from human-readable text on clipboard"""
        # No parameters to paste
        pass

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        # No parameters to paste
        pass

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        return ""

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Invert the frame

        Args:
            frame: Input frame (BGR format)
            face_mask: Optional face detection mask (unused)

        Returns:
            Inverted frame
        """
        if not self.enabled.get():
            return frame

        return 255 - frame

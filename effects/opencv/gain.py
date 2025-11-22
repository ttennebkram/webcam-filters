"""
Gain effect - adjusts image brightness with logarithmic scaling.
"""

import math
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class GainEffect(BaseUIEffect):
    """Effect that adjusts image brightness using a gain multiplier with logarithmic scaling"""

    def __init__(self, width: int, height: int, root=None, **kwargs):
        """Initialize the gain effect

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            root: Tkinter root window
        """
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        # Use logarithmic slider variable (-1 to 1 maps to 0.1x to 10x)
        self.gain_slider_var = tk.DoubleVar(value=0.0)  # log10(1.0) = 0
        self.gain_var = tk.DoubleVar(value=1.0)  # Actual gain value
        self.gain_display_var = tk.StringVar(value="1.00x")

        # Set up trace to update gain when slider changes
        self.gain_slider_var.trace_add('write', self._on_slider_change)

    def _on_slider_change(self, *args):
        """Convert logarithmic slider value to actual gain"""
        log_value = self.gain_slider_var.get()
        actual_gain = 10 ** log_value
        self.gain_var.set(actual_gain)
        self.gain_display_var.set(f"{actual_gain:.2f}x")

    @classmethod
    def get_name(cls) -> str:
        return "Gain"

    @classmethod
    def get_description(cls) -> str:
        return "Adjusts image brightness with logarithmic gain control (0.1x to 10x)"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.multiply(src, gain)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        # Custom slider with tick marks - we'll build this in the control panel
        return []

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'gain': self.gain_var.get()
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

        # Add custom gain slider with tick marks to the center frame
        if hasattr(self._effect_form, '_center_frame'):
            self._create_gain_slider(self._effect_form._center_frame, mode)

        return self.control_panel

    def _create_gain_slider(self, parent, mode):
        """Create the gain slider with logarithmic scale and tick marks"""
        # Container for slider and tick marks
        gain_container = ttk.Frame(parent)
        gain_container.pack(fill='x', pady=(5, 0))

        # Label (both modes)
        ttk.Label(gain_container, text="Gain: 0.1x to 10x").pack(anchor='w')

        if mode == 'view':
            # View mode - just show the value as text
            view_frame = ttk.Frame(gain_container)
            view_frame.pack(fill='x')
            ttk.Label(view_frame, textvariable=self.gain_display_var).pack(side='left', pady=2)
        else:
            # Edit mode - Frame for slider and display
            slider_frame = ttk.Frame(gain_container)
            slider_frame.pack(fill='x')

            # Editable slider
            self._gain_slider = ttk.Scale(
                slider_frame, from_=-1, to=1,
                variable=self.gain_slider_var,
                orient='horizontal'
            )
            self._gain_slider.pack(side='left', fill='x', expand=True, padx=(0, 5))

            # Display label
            ttk.Label(slider_frame, textvariable=self.gain_display_var, width=7).pack(side='left')

        # Add tick marks below (both modes for consistent height)
        # Use ttk.Frame which properly inherits theme background
        tick_frame = ttk.Frame(gain_container, height=8)
        tick_frame.pack(fill='x', padx=(0, 52))
        tick_frame.pack_propagate(False)

        self._tick_canvas = tk.Canvas(
            tick_frame, height=8,
            highlightthickness=0, bd=0,
            bg='systemTransparent'  # macOS transparent
        )
        self._tick_canvas.pack(fill='both', expand=True)

        self._tick_canvas.bind('<Configure>', lambda e: self._draw_tick_marks())
        self._draw_tick_marks()

    def _draw_tick_marks(self):
        """Draw tick marks at logarithmic positions"""
        if not hasattr(self, '_tick_canvas'):
            return

        self._tick_canvas.delete('all')
        canvas_width = self._tick_canvas.winfo_width()
        if canvas_width <= 1:
            # Not yet rendered, try again later
            self._tick_canvas.after(100, self._draw_tick_marks)
            return

        try:
            slider_width = self._gain_slider.winfo_width()
            # ttk.Scale has internal padding
            trough_padding = 9
            trough_width = slider_width - (2 * trough_padding)

            # Tick positions at logarithmic intervals
            ticks = []

            # Add ticks for 0.1x to 0.9x
            for i in range(1, 10):
                gain = i / 10.0
                log_pos = math.log10(gain)
                ticks.append((log_pos, 3))

            # Large tick at 1x
            ticks.append((0.0, 6))

            # Add ticks for 2x to 10x
            for i in range(2, 11):
                gain = float(i)
                log_pos = math.log10(gain)
                ticks.append((log_pos, 3))

            for log_pos, height in ticks:
                # Map from -1..1 to the trough area
                x = int(trough_padding + (log_pos + 1) / 2 * trough_width)
                self._tick_canvas.create_line(x, 0, x, height, fill='gray40', width=1)
        except:
            # If we can't get dimensions yet, try again
            self._tick_canvas.after(100, self._draw_tick_marks)

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

        # Add custom gain slider
        if hasattr(self._effect_form, '_center_frame'):
            self._create_gain_slider(self._effect_form._center_frame, self._current_mode)

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())
        lines.append(f"Gain: {self.gain_var.get():.2f}x")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'gain': self.gain_var.get()
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
                if line.startswith('Gain:'):
                    value_str = line.split(':', 1)[1].strip()
                    # Remove 'x' suffix if present
                    value_str = value_str.rstrip('x').strip()
                    gain = float(value_str)
                    # Clamp to valid range
                    gain = max(0.1, min(10.0, gain))
                    self.gain_var.set(gain)
                    # Update slider to match
                    self.gain_slider_var.set(math.log10(gain))
        except:
            pass

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'gain' in data:
                gain = float(data['gain'])
                # Clamp to valid range
                gain = max(0.1, min(10.0, gain))
                self.gain_var.set(gain)
                # Update slider to match
                self.gain_slider_var.set(math.log10(gain))
        except:
            pass

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        return f"Gain: {self.gain_var.get():.2f}x"

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply gain to the frame

        Args:
            frame: Input frame (BGR format)
            face_mask: Optional face detection mask (unused)

        Returns:
            Frame with gain applied
        """
        if not self.enabled.get():
            return frame

        gain = self.gain_var.get()

        # Apply gain using cv2.multiply with saturation
        result = cv2.multiply(frame.astype(np.float32), gain)

        # Clip to valid range and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

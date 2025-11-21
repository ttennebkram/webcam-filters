"""
Bit Plane Decomposition effect using OpenCV.

Decomposes grayscale image into 8 bit planes with individual gain controls.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class BitPlanesEffect(BaseUIEffect):
    """Decompose grayscale into bit planes with individual gain controls"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Bit plane controls (8 bit planes: 7 MSB down to 0 LSB)
        self.bitplane_enable = []
        self.bitplane_gain = []
        self.bitplane_gain_slider = []
        for i in range(8):
            self.bitplane_enable.append(tk.BooleanVar(value=True))
            self.bitplane_gain.append(tk.DoubleVar(value=1.0))
            self.bitplane_gain_slider.append(tk.DoubleVar(value=0.0))  # Log scale: 0 = 1x

        # "All" controls for bit planes
        self.bitplane_all_enable = tk.BooleanVar(value=True)
        self.bitplane_all_gain = tk.DoubleVar(value=1.0)
        self.bitplane_all_gain_slider = tk.DoubleVar(value=0.0)

    @classmethod
    def get_name(cls) -> str:
        return "Bit Planes Grayscale"

    @classmethod
    def get_description(cls) -> str:
        return "Select and adjust gain on individual grayscale bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    @classmethod
    def get_method_signature(cls) -> str:
        return "Bit plane decomposition with gain"

    def get_form_schema(self):
        """Return empty schema - bit planes uses custom rendering"""
        return []

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        data = {}
        for i in range(8):
            data[f'enable_{i}'] = self.bitplane_enable[i].get()
            data[f'gain_{i}'] = self.bitplane_gain[i].get()
        return data

    def get_view_mode_summary(self) -> str:
        """Return a human-readable summary of enabled bits and gains for view mode"""
        enabled_bits = []
        for i in range(8):
            if self.bitplane_enable[i].get():
                gain = self.bitplane_gain[i].get()
                bit_num = 7 - i  # Convert index to bit number
                if abs(gain - 1.0) < 0.01:
                    enabled_bits.append(str(bit_num))
                else:
                    enabled_bits.append(f"{bit_num}({gain:.1f}x)")

        if enabled_bits:
            return f"Bits: {', '.join(enabled_bits)}"
        else:
            return "Bits: none"

    def get_pipeline_params(self) -> dict:
        """Return custom parameters for pipeline saving"""
        params = {}
        for i in range(8):
            params[f'enable_{i}'] = self.bitplane_enable[i].get()
            params[f'gain_{i}'] = self.bitplane_gain[i].get()
        return params

    def set_pipeline_params(self, params: dict):
        """Restore custom parameters from pipeline loading"""
        for i in range(8):
            enable_key = f'enable_{i}'
            gain_key = f'gain_{i}'
            if enable_key in params:
                self.bitplane_enable[i].set(params[enable_key])
            if gain_key in params:
                gain = params[gain_key]
                self.bitplane_gain[i].set(gain)
                self.bitplane_gain_slider[i].set(self._gain_to_slider(gain))

    def _slider_to_gain(self, slider_val):
        """Convert slider value (-1 to 1) to gain (0.1x to 10x)"""
        return 10 ** slider_val

    def _gain_to_slider(self, gain):
        """Convert gain to slider value"""
        import math
        if gain <= 0:
            return -1
        return math.log10(gain)

    def create_control_panel(self, parent, mode='view'):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)
        self._control_parent = parent
        self._current_mode = mode

        # Create the EffectForm with empty subform (we'll add custom content)
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

        # In edit mode, add the custom bit plane table to the center column
        if mode == 'edit':
            # Find the center frame in the EffectForm and add our custom table
            self._add_bitplane_table(self._effect_form._center_frame)
        else:
            # In view mode, show the bits summary
            self._add_view_summary(self._effect_form._center_frame)

        # Force geometry recalculation to ensure proper sizing
        self.control_panel.update_idletasks()

        return self.control_panel

    def _add_bitplane_table(self, parent):
        """Add the bit plane table to the given parent frame"""
        # Create table for bit planes
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='x', pady=(5, 0))

        # Header row
        ttk.Label(table_frame, text="Bit").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        ttk.Label(table_frame, text="On").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Gain (0.1x to 10x)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)

        # "All" row at the top
        ttk.Label(table_frame, text="All", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=0, padx=5, pady=3, sticky='e')

        def on_bitplane_all_enable_change(*args):
            enabled = self.bitplane_all_enable.get()
            for i in range(8):
                self.bitplane_enable[i].set(enabled)

        self.bitplane_all_enable.trace_add("write", on_bitplane_all_enable_change)
        ttk.Checkbutton(table_frame, variable=self.bitplane_all_enable).grid(row=1, column=1, padx=5, pady=3)

        def update_all_bitplane_gain(slider_val):
            gain = self._slider_to_gain(float(slider_val))
            self.bitplane_all_gain.set(gain)
            for i in range(8):
                self.bitplane_gain[i].set(gain)
                self.bitplane_gain_slider[i].set(slider_val)

        all_gain_slider = ttk.Scale(table_frame, from_=-1, to=1, variable=self.bitplane_all_gain_slider, orient='horizontal',
                                    command=lambda v: update_all_bitplane_gain(v))
        all_gain_slider.grid(row=1, column=2, padx=5, pady=3, sticky='ew')

        # Display label for gain value
        self.all_gain_label = ttk.Label(table_frame, text="1.00x", width=6)
        self.all_gain_label.grid(row=1, column=3, padx=(2, 5), pady=3)

        def update_all_gain_label(*args):
            gain = self.bitplane_all_gain.get()
            self.all_gain_label.config(text=f"{gain:.2f}x")

        self.bitplane_all_gain.trace_add("write", update_all_gain_label)

        # Separator line after "All" row
        sep_frame = ttk.Frame(table_frame, height=2, relief='sunken')
        sep_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(5, 5))

        # Create 8 rows for bit planes
        bit_labels = ["7", "6", "5", "4", "3", "2", "1", "0"]
        self.gain_labels = []

        for i, label in enumerate(bit_labels):
            row = i + 3  # Start at row 3 (after header, All row, and separator)

            ttk.Label(table_frame, text=label).grid(row=row, column=0, padx=5, pady=2, sticky='e')
            ttk.Checkbutton(table_frame, variable=self.bitplane_enable[i]).grid(row=row, column=1, padx=5, pady=2)

            def update_gain(slider_val, idx=i):
                gain = self._slider_to_gain(float(slider_val))
                self.bitplane_gain[idx].set(gain)

            gain_slider = ttk.Scale(table_frame, from_=-1, to=1, variable=self.bitplane_gain_slider[i], orient='horizontal',
                                    command=lambda v, idx=i: update_gain(v, idx))
            gain_slider.grid(row=row, column=2, padx=5, pady=2, sticky='ew')

            # Gain value label
            gain_label = ttk.Label(table_frame, text="1.00x", width=6)
            gain_label.grid(row=row, column=3, padx=(2, 5), pady=2)
            self.gain_labels.append(gain_label)

            def update_gain_label(var_name, index, mode, idx=i):
                gain = self.bitplane_gain[idx].get()
                self.gain_labels[idx].config(text=f"{gain:.2f}x")

            self.bitplane_gain[i].trace_add("write", update_gain_label)

        table_frame.columnconfigure(2, weight=1)

    def _add_view_summary(self, parent):
        """Add the bits summary label for view mode"""
        # Build the bits string without the "Bits: " prefix since we'll use a proper label
        enabled_bits = []
        for i in range(8):
            if self.bitplane_enable[i].get():
                gain = self.bitplane_gain[i].get()
                bit_num = 7 - i  # Convert index to bit number
                if abs(gain - 1.0) < 0.01:
                    enabled_bits.append(str(bit_num))
                else:
                    enabled_bits.append(f"{bit_num}({gain:.1f}x)")

        if enabled_bits:
            bits_value = ', '.join(enabled_bits)
        else:
            bits_value = "none"

        # Create frame for the label row
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill='x', pady=(5, 0))

        # Right-justified "Bits:" label
        ttk.Label(summary_frame, text="Bits:").grid(row=0, column=0, sticky='e', padx=(5, 10), pady=4)
        # Value label
        ttk.Label(summary_frame, text=bits_value).grid(row=0, column=1, sticky='w', pady=4)

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
            self._add_bitplane_table(self._effect_form._center_frame)
        else:
            self._add_view_summary(self._effect_form._center_frame)

        # Force geometry recalculation to ensure proper sizing
        self.control_panel.update_idletasks()

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())
        lines.append(self.get_view_mode_summary())

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
        }
        for i in range(8):
            data[f'enable_{i}'] = self.bitplane_enable[i].get()
            data[f'gain_{i}'] = self.bitplane_gain[i].get()

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
            # Parse "Bits: 7, 6(2.0x), 5, ..." format
            if 'Bits:' in text:
                bits_part = text.split('Bits:')[1].strip().split('\n')[0]
                # Reset all bits to disabled
                for i in range(8):
                    self.bitplane_enable[i].set(False)
                    self.bitplane_gain[i].set(1.0)
                    self.bitplane_gain_slider[i].set(0.0)

                if bits_part.lower() != 'none':
                    for bit_spec in bits_part.split(','):
                        bit_spec = bit_spec.strip()
                        if '(' in bit_spec:
                            # Has gain: "7(2.0x)"
                            bit_num = int(bit_spec.split('(')[0])
                            gain_str = bit_spec.split('(')[1].rstrip('x)')
                            gain = float(gain_str)
                        else:
                            bit_num = int(bit_spec)
                            gain = 1.0

                        # Convert bit number to index (7 -> 0, 0 -> 7)
                        idx = 7 - bit_num
                        if 0 <= idx < 8:
                            self.bitplane_enable[idx].set(True)
                            self.bitplane_gain[idx].set(gain)
                            self.bitplane_gain_slider[idx].set(self._gain_to_slider(gain))
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            for i in range(8):
                if f'enable_{i}' in data:
                    self.bitplane_enable[i].set(data[f'enable_{i}'])
                if f'gain_{i}' in data:
                    gain = data[f'gain_{i}']
                    self.bitplane_gain[i].set(gain)
                    self.bitplane_gain_slider[i].set(self._gain_to_slider(gain))
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply bit plane decomposition with gain to the frame"""
        if not self.enabled.get():
            return frame

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Initialize result as float for accumulation
        result = np.zeros_like(gray, dtype=np.float32)

        # Process each bit plane
        for i in range(8):
            if not self.bitplane_enable[i].get():
                continue

            # Extract bit plane (bit 7-i, since i=0 is MSB)
            bit_index = 7 - i
            bit_plane = (gray >> bit_index) & 1

            # Scale to original bit weight
            bit_value = bit_plane.astype(np.float32) * (1 << bit_index)

            # Apply gain
            gain = self.bitplane_gain[i].get()
            bit_value *= gain

            # Accumulate
            result += bit_value

        # Clip to valid range
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Convert back to BGR for display
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

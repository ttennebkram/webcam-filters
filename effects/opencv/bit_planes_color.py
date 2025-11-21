"""
Color Bit Plane Decomposition effect using OpenCV.

Decomposes RGB image into 24 bit planes (8 per channel) with individual gain controls.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import json
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm


class ColorBitPlanesEffect(BaseUIEffect):
    """Decompose RGB into bit planes with individual gain controls per channel"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)

        # Color bit plane controls (3 colors × 8 bit planes each)
        self.color_bitplane_enable = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_gain = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_gain_slider = {'red': [], 'green': [], 'blue': []}

        for color in ['red', 'green', 'blue']:
            for i in range(8):
                self.color_bitplane_enable[color].append(tk.BooleanVar(value=True))
                self.color_bitplane_gain[color].append(tk.DoubleVar(value=1.0))
                self.color_bitplane_gain_slider[color].append(tk.DoubleVar(value=0.0))

        # "All" controls for each color
        self.color_bitplane_all_enable = {
            'red': tk.BooleanVar(value=True),
            'green': tk.BooleanVar(value=True),
            'blue': tk.BooleanVar(value=True)
        }
        self.color_bitplane_all_gain = {
            'red': tk.DoubleVar(value=1.0),
            'green': tk.DoubleVar(value=1.0),
            'blue': tk.DoubleVar(value=1.0)
        }
        self.color_bitplane_all_gain_slider = {
            'red': tk.DoubleVar(value=0.0),
            'green': tk.DoubleVar(value=0.0),
            'blue': tk.DoubleVar(value=0.0)
        }

        # Tab management
        self.color_bp_tab_buttons = {}
        self.color_bp_tab_frames = {}
        self.color_bp_selected_tab = None

        # Flag to prevent trace callbacks during programmatic changes
        self._restoring = False

    @classmethod
    def get_name(cls) -> str:
        return "Bit Planes Color"

    @classmethod
    def get_description(cls) -> str:
        return "Select and adjust gain on individual RGB bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    @classmethod
    def get_method_signature(cls) -> str:
        return "Bit plane decomposition with gain (RGB)"

    def get_form_schema(self):
        """Return empty schema - color bit planes uses custom rendering"""
        return []

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        data = {}
        for color in ['red', 'green', 'blue']:
            for i in range(8):
                data[f'{color}_enable_{i}'] = self.color_bitplane_enable[color][i].get()
                data[f'{color}_gain_{i}'] = self.color_bitplane_gain[color][i].get()
        return data

    def restore_state(self):
        """Restore tk.Variable values from the last snapshot.

        Override to set _restoring flag to prevent traces from overwriting values.
        """
        self._restoring = True
        super().restore_state()
        self._restoring = False

    def get_view_mode_summary(self) -> str:
        """Return a human-readable summary of enabled channels and gains for view mode"""
        lines = []
        for color in ['red', 'green', 'blue']:
            enabled_bits = []
            for i in range(8):
                if self.color_bitplane_enable[color][i].get():
                    gain = self.color_bitplane_gain[color][i].get()
                    bit_num = 7 - i  # Convert index to bit number
                    if abs(gain - 1.0) < 0.01:
                        enabled_bits.append(str(bit_num))
                    else:
                        enabled_bits.append(f"{bit_num}({gain:.1f}x)")

            if enabled_bits:
                lines.append(f"{color.capitalize()}: {', '.join(enabled_bits)}")
            else:
                lines.append(f"{color.capitalize()}: none")

        return '\n'.join(lines)

    def get_pipeline_params(self) -> dict:
        """Return custom parameters for pipeline saving"""
        params = {}
        for color in ['red', 'green', 'blue']:
            for i in range(8):
                params[f'{color}_enable_{i}'] = self.color_bitplane_enable[color][i].get()
                params[f'{color}_gain_{i}'] = self.color_bitplane_gain[color][i].get()
        return params

    def set_pipeline_params(self, params: dict):
        """Restore custom parameters from pipeline loading"""
        for color in ['red', 'green', 'blue']:
            for i in range(8):
                enable_key = f'{color}_enable_{i}'
                gain_key = f'{color}_gain_{i}'
                if enable_key in params:
                    self.color_bitplane_enable[color][i].set(params[enable_key])
                if gain_key in params:
                    gain = params[gain_key]
                    self.color_bitplane_gain[color][i].set(gain)
                    self.color_bitplane_gain_slider[color][i].set(self._gain_to_slider(gain))

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

        # In edit mode, add the custom tabbed color panels to the center column
        if mode == 'edit':
            self._add_color_tabs(self._effect_form._center_frame)
        else:
            # In view mode, show the bits summary
            self._add_view_summary(self._effect_form._center_frame)

        # Force geometry recalculation to ensure proper sizing
        self.control_panel.update_idletasks()

        return self.control_panel

    def _add_view_summary(self, parent):
        """Add the bits summary labels for view mode"""
        summary_frame = ttk.Frame(parent)
        summary_frame.pack(fill='x', pady=(5, 0))

        for row_idx, color in enumerate(['red', 'green', 'blue']):
            # Build the bits string for this color
            enabled_bits = []
            for i in range(8):
                if self.color_bitplane_enable[color][i].get():
                    gain = self.color_bitplane_gain[color][i].get()
                    bit_num = 7 - i  # Convert index to bit number
                    if abs(gain - 1.0) < 0.01:
                        enabled_bits.append(str(bit_num))
                    else:
                        enabled_bits.append(f"{bit_num}({gain:.1f}x)")

            if enabled_bits:
                bits_value = ', '.join(enabled_bits)
            else:
                bits_value = "none"

            # Right-justified color label
            ttk.Label(summary_frame, text=f"{color.capitalize()}:").grid(
                row=row_idx, column=0, sticky='e', padx=(5, 10), pady=4
            )
            # Value label
            ttk.Label(summary_frame, text=bits_value).grid(
                row=row_idx, column=1, sticky='w', pady=4
            )

    def _add_color_tabs(self, parent):
        """Add the tabbed color panels to the given parent frame"""
        # Create custom tab bar
        tab_bar = ttk.Frame(parent)
        tab_bar.pack(fill='x', pady=(5, 0))

        tab_buttons_frame = tk.Frame(tab_bar, bg='gray85')
        tab_buttons_frame.pack(side='left', fill='x')

        self.color_bp_selected_tab = tk.StringVar(value='red')

        tab_content_container = ttk.Frame(parent, relief='solid', borderwidth=1)
        tab_content_container.pack(fill='both', expand=True)

        style = ttk.Style()

        # Create tabs for Red, Green, Blue
        for color_name, color_fg in [('Red', 'red'), ('Green', 'green'), ('Blue', 'DeepSkyBlue')]:
            color_key = color_name.lower()

            # Create colored tab button
            tab_btn = tk.Label(tab_buttons_frame, text=color_name, foreground=color_fg,
                              relief='raised', borderwidth=1, padx=10, pady=5,
                              font=('TkDefaultFont', 9, 'bold'), bg='gray85')
            tab_btn.pack(side='left', padx=(0, 2))
            tab_btn.bind('<Button-1>', lambda e, c=color_key: self._switch_color_tab(c))
            self.color_bp_tab_buttons[color_key] = tab_btn

            # Create frame for this tab's content
            tab_frame = ttk.Frame(tab_content_container)
            self.color_bp_tab_frames[color_key] = tab_frame

            # Create table for this color's bit planes
            table_frame = ttk.Frame(tab_frame)
            table_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Create style for colored labels
            style_name = f"{color_key}_label.TLabel"
            style.configure(style_name, foreground=color_fg)

            # Header row
            ttk.Label(table_frame, text="Bit", style=style_name).grid(row=0, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(table_frame, text="On", style=style_name).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text="Gain (0.1x to 10x)", style=style_name).grid(row=0, column=2, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)

            # "All" row at the top
            all_style_name = f"{color_key}_all_label.TLabel"
            style.configure(all_style_name, foreground=color_fg, font=('TkDefaultFont', 10, 'bold'))
            ttk.Label(table_frame, text="All", style=all_style_name).grid(row=1, column=0, padx=5, pady=3, sticky='e')

            # All enable checkbox
            def on_color_all_enable_change(var_name, index, mode, c=color_key):
                # Skip during restore to prevent overwriting individual values
                if self._restoring:
                    return
                enabled = self.color_bitplane_all_enable[c].get()
                for i in range(8):
                    self.color_bitplane_enable[c][i].set(enabled)

            self.color_bitplane_all_enable[color_key].trace_add("write", on_color_all_enable_change)
            ttk.Checkbutton(table_frame, variable=self.color_bitplane_all_enable[color_key]).grid(row=1, column=1, padx=5, pady=3)

            # All gain slider
            def update_all_color_gain(slider_val, c=color_key):
                gain = self._slider_to_gain(float(slider_val))
                self.color_bitplane_all_gain[c].set(gain)
                for i in range(8):
                    self.color_bitplane_gain[c][i].set(gain)
                    self.color_bitplane_gain_slider[c][i].set(slider_val)

            all_gain_slider = ttk.Scale(table_frame, from_=-1, to=1,
                                        variable=self.color_bitplane_all_gain_slider[color_key],
                                        orient='horizontal',
                                        command=lambda v, c=color_key: update_all_color_gain(v, c))
            all_gain_slider.grid(row=1, column=2, padx=5, pady=3, sticky='ew')

            # All gain label
            all_gain_label = ttk.Label(table_frame, text="1.00x", width=6)
            all_gain_label.grid(row=1, column=3, padx=(2, 5), pady=3)

            def update_all_color_gain_label(var_name, index, mode, c=color_key, lbl=all_gain_label):
                try:
                    if lbl.winfo_exists():
                        gain = self.color_bitplane_all_gain[c].get()
                        lbl.config(text=f"{gain:.2f}x")
                except:
                    pass

            self.color_bitplane_all_gain[color_key].trace_add("write", update_all_color_gain_label)

            # Separator line after "All" row
            sep_frame = ttk.Frame(table_frame, height=2, relief='sunken')
            sep_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(5, 5))

            # Create 8 rows for bit planes
            bit_labels = ["7", "6", "5", "4", "3", "2", "1", "0"]
            gain_labels = []

            for i, label in enumerate(bit_labels):
                row = i + 3  # Start at row 3

                ttk.Label(table_frame, text=label, style=style_name).grid(row=row, column=0, padx=5, pady=2, sticky='e')
                ttk.Checkbutton(table_frame, variable=self.color_bitplane_enable[color_key][i]).grid(row=row, column=1, padx=5, pady=2)

                def update_gain(slider_val, c=color_key, idx=i):
                    gain = self._slider_to_gain(float(slider_val))
                    self.color_bitplane_gain[c][idx].set(gain)

                gain_slider = ttk.Scale(table_frame, from_=-1, to=1,
                                        variable=self.color_bitplane_gain_slider[color_key][i],
                                        orient='horizontal',
                                        command=lambda v, c=color_key, idx=i: update_gain(v, c, idx))
                gain_slider.grid(row=row, column=2, padx=5, pady=2, sticky='ew')

                # Gain value label
                gain_label = ttk.Label(table_frame, text="1.00x", width=6)
                gain_label.grid(row=row, column=3, padx=(2, 5), pady=2)
                gain_labels.append(gain_label)

                def update_gain_label(var_name, index, mode, c=color_key, idx=i, lbl=gain_label):
                    try:
                        if lbl.winfo_exists():
                            gain = self.color_bitplane_gain[c][idx].get()
                            lbl.config(text=f"{gain:.2f}x")
                    except:
                        pass  # Widget destroyed during hot-swap

                self.color_bitplane_gain[color_key][i].trace_add("write", update_gain_label)

            table_frame.columnconfigure(2, weight=1)

        # Show initial tab
        self._switch_color_tab('red')

    def _switch_color_tab(self, color_key):
        """Switch to the specified color tab"""
        self.color_bp_selected_tab.set(color_key)

        # Hide all tab frames
        for key, frame in self.color_bp_tab_frames.items():
            frame.pack_forget()

        # Show selected tab frame
        self.color_bp_tab_frames[color_key].pack(fill='both', expand=True)

        # Update tab button appearances
        for key, btn in self.color_bp_tab_buttons.items():
            if key == color_key:
                btn.config(relief='sunken', bg='white')
            else:
                btn.config(relief='raised', bg='gray85')

    def _toggle_mode(self):
        """Toggle between edit and view modes"""
        self._current_mode = 'view' if self._current_mode == 'edit' else 'edit'

        # Notify pipeline when switching to edit mode
        if self._current_mode == 'edit' and hasattr(self, '_on_edit') and self._on_edit:
            self._on_edit()

        # Re-render the entire control panel
        for child in self.control_panel.winfo_children():
            child.destroy()

        # Reset tab management for fresh rendering
        self.color_bp_tab_buttons = {}
        self.color_bp_tab_frames = {}

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
            self._add_color_tabs(self._effect_form._center_frame)
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
        for color in ['red', 'green', 'blue']:
            for i in range(8):
                bit_num = 7 - i  # Convert index to bit number
                data[f'{color}_bit_{bit_num}_enabled'] = self.color_bitplane_enable[color][i].get()
                data[f'{color}_bit_{bit_num}_gain'] = self.color_bitplane_gain[color][i].get()

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

                    # Parse color lines like "Red: 7, 6(2.0x), 5, ..."
                    if key in ['red', 'green', 'blue']:
                        # Reset all bits for this color
                        for i in range(8):
                            self.color_bitplane_enable[key][i].set(False)
                            self.color_bitplane_gain[key][i].set(1.0)
                            self.color_bitplane_gain_slider[key][i].set(0.0)

                        if value.lower() != 'none':
                            for bit_spec in value.split(','):
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
                                    self.color_bitplane_enable[key][idx].set(True)
                                    self.color_bitplane_gain[key][idx].set(gain)
                                    self.color_bitplane_gain_slider[key][idx].set(self._gain_to_slider(gain))
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            for color in ['red', 'green', 'blue']:
                for i in range(8):
                    bit_num = 7 - i  # Convert index to bit number
                    if f'{color}_bit_{bit_num}_enabled' in data:
                        self.color_bitplane_enable[color][i].set(data[f'{color}_bit_{bit_num}_enabled'])
                    if f'{color}_bit_{bit_num}_gain' in data:
                        gain = data[f'{color}_bit_{bit_num}_gain']
                        self.color_bitplane_gain[color][i].set(gain)
                        self.color_bitplane_gain_slider[color][i].set(self._gain_to_slider(gain))
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply color bit plane decomposition with gain to the frame"""
        if not self.enabled.get():
            return frame

        # Ensure we have a color image
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Split into BGR channels
        b, g, r = cv2.split(frame)

        # Process each channel
        channels = {'red': r, 'green': g, 'blue': b}
        results = {}

        for color_key, channel in channels.items():
            # Initialize result as float for accumulation
            result = np.zeros_like(channel, dtype=np.float32)

            # Process each bit plane
            for i in range(8):
                if not self.color_bitplane_enable[color_key][i].get():
                    continue

                # Extract bit plane (bit 7-i, since i=0 is MSB)
                bit_index = 7 - i
                bit_plane = (channel >> bit_index) & 1

                # Scale to original bit weight
                bit_value = bit_plane.astype(np.float32) * (1 << bit_index)

                # Apply gain
                gain = self.color_bitplane_gain[color_key][i].get()
                bit_value *= gain

                # Accumulate
                result += bit_value

            # Clip to valid range
            results[color_key] = np.clip(result, 0, 255).astype(np.uint8)

        # Merge channels back
        result = cv2.merge([results['blue'], results['green'], results['red']])

        return result

"""
Color Bit Plane Decomposition effect using OpenCV.

Decomposes RGB image into 24 bit planes (8 per channel) with individual gain controls.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


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

    @classmethod
    def get_name(cls) -> str:
        return "Color Bit Plane Selector"

    @classmethod
    def get_description(cls) -> str:
        return "Select and adjust gain on individual RGB bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

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
                lines.append(f"{color.capitalize()}: bits {', '.join(enabled_bits)}")
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

    def create_control_panel(self, parent):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)

        padding = {'padx': 10, 'pady': 5}

        # Header section (skip if in pipeline - LabelFrame already shows name)
        if not getattr(self, '_in_pipeline', False):
            header_frame = ttk.Frame(self.control_panel)
            header_frame.pack(fill='x', **padding)

            # Title
            title_label = ttk.Label(
                header_frame,
                text="Color Bit Plane Selector",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Description
            desc_label = ttk.Label(
                header_frame,
                text="Select and adjust gain on RGB bit planes",
                font=('TkFixedFont', 12)
            )
            desc_label.pack(anchor='w', pady=(2, 2))

        # Main frame with two columns
        main_frame = ttk.Frame(self.control_panel)
        main_frame.pack(fill='x', **padding)

        # Left column - Enabled checkbox
        left_column = ttk.Frame(main_frame)
        left_column.pack(side='left', fill='y', padx=(0, 15))

        ttk.Frame(left_column).pack(expand=True)
        enabled_cb = ttk.Checkbutton(
            left_column,
            text="Enabled",
            variable=self.enabled
        )
        enabled_cb.pack()
        ttk.Frame(left_column).pack(expand=True)

        # Right column - tabbed color panels
        right_column = ttk.Frame(main_frame)
        right_column.pack(side='left', fill='both', expand=True)

        # Create custom tab bar
        tab_bar = ttk.Frame(right_column)
        tab_bar.pack(fill='x')

        tab_buttons_frame = tk.Frame(tab_bar, bg='gray85')
        tab_buttons_frame.pack(side='left', fill='x')

        self.color_bp_selected_tab = tk.StringVar(value='red')

        tab_content_container = ttk.Frame(right_column, relief='solid', borderwidth=1)
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
            ttk.Label(table_frame, text="Enabled", style=style_name).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text="Gain (0.1x to 10x)", style=style_name).grid(row=0, column=2, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)

            # "All" row at the top
            all_style_name = f"{color_key}_all_label.TLabel"
            style.configure(all_style_name, foreground=color_fg, font=('TkDefaultFont', 14, 'bold'))
            ttk.Label(table_frame, text="All", style=all_style_name).grid(row=1, column=0, padx=5, pady=5, sticky='e')

            # All enable checkbox
            def on_color_all_enable_change(var_name, index, mode, c=color_key):
                enabled = self.color_bitplane_all_enable[c].get()
                for i in range(8):
                    self.color_bitplane_enable[c][i].set(enabled)

            self.color_bitplane_all_enable[color_key].trace_add("write", on_color_all_enable_change)
            ttk.Checkbutton(table_frame, variable=self.color_bitplane_all_enable[color_key]).grid(row=1, column=1, padx=5, pady=5)

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
            all_gain_slider.grid(row=1, column=2, padx=5, pady=5, sticky='ew')

            # All gain label
            all_gain_label = ttk.Label(table_frame, text="1.00x", width=6)
            all_gain_label.grid(row=1, column=3, padx=(2, 10), pady=5)

            def update_all_color_gain_label(var_name, index, mode, c=color_key, lbl=all_gain_label):
                gain = self.color_bitplane_all_gain[c].get()
                lbl.config(text=f"{gain:.2f}x")

            self.color_bitplane_all_gain[color_key].trace_add("write", update_all_color_gain_label)

            # Separator line after "All" row
            sep_frame = ttk.Frame(table_frame, height=2, relief='sunken')
            sep_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(10, 10))

            # Create 8 rows for bit planes
            bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]
            gain_labels = []

            for i, label in enumerate(bit_labels):
                row = i + 3  # Start at row 3

                ttk.Label(table_frame, text=label, style=style_name).grid(row=row, column=0, padx=5, pady=5, sticky='e')
                ttk.Checkbutton(table_frame, variable=self.color_bitplane_enable[color_key][i]).grid(row=row, column=1, padx=5, pady=5)

                def update_gain(slider_val, c=color_key, idx=i):
                    gain = self._slider_to_gain(float(slider_val))
                    self.color_bitplane_gain[c][idx].set(gain)

                gain_slider = ttk.Scale(table_frame, from_=-1, to=1,
                                        variable=self.color_bitplane_gain_slider[color_key][i],
                                        orient='horizontal',
                                        command=lambda v, c=color_key, idx=i: update_gain(v, c, idx))
                gain_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')

                # Gain value label
                gain_label = ttk.Label(table_frame, text="1.00x", width=6)
                gain_label.grid(row=row, column=3, padx=(2, 10), pady=5)
                gain_labels.append(gain_label)

                def update_gain_label(var_name, index, mode, c=color_key, idx=i, lbl=gain_label):
                    gain = self.color_bitplane_gain[c][idx].get()
                    lbl.config(text=f"{gain:.2f}x")

                self.color_bitplane_gain[color_key][i].trace_add("write", update_gain_label)

            table_frame.columnconfigure(2, weight=1)

        # Show initial tab
        self._switch_color_tab('red')

        return self.control_panel

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

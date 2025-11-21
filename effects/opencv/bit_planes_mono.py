"""
Bit Plane Decomposition effect using OpenCV.

Decomposes grayscale image into 8 bit planes with individual gain controls.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect


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
        return "Grayscale Bit Plane Selector"

    @classmethod
    def get_description(cls) -> str:
        return "Select and adjust gain on individual grayscale bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

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
                text="Grayscale Bit Plane Selector",
                font=('TkDefaultFont', 14, 'bold')
            )
            title_label.pack(anchor='w')

            # Description
            desc_label = ttk.Label(
                header_frame,
                text="Select and adjust gain on bit planes",
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

        # Right column - bit plane table
        right_column = ttk.Frame(main_frame)
        right_column.pack(side='left', fill='both', expand=True)

        # Create table for bit planes
        table_frame = ttk.Frame(right_column)
        table_frame.pack(fill='x')

        # Header row
        ttk.Label(table_frame, text="Bit").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        ttk.Label(table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Gain (0.1x to 10x)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)

        # "All" row at the top
        ttk.Label(table_frame, text="All", font=('TkDefaultFont', 14, 'bold')).grid(row=1, column=0, padx=5, pady=5, sticky='e')

        def on_bitplane_all_enable_change(*args):
            enabled = self.bitplane_all_enable.get()
            for i in range(8):
                self.bitplane_enable[i].set(enabled)

        self.bitplane_all_enable.trace_add("write", on_bitplane_all_enable_change)
        ttk.Checkbutton(table_frame, variable=self.bitplane_all_enable).grid(row=1, column=1, padx=5, pady=5)

        def update_all_bitplane_gain(slider_val):
            gain = self._slider_to_gain(float(slider_val))
            self.bitplane_all_gain.set(gain)
            for i in range(8):
                self.bitplane_gain[i].set(gain)
                self.bitplane_gain_slider[i].set(slider_val)

        all_gain_slider = ttk.Scale(table_frame, from_=-1, to=1, variable=self.bitplane_all_gain_slider, orient='horizontal',
                                    command=lambda v: update_all_bitplane_gain(v))
        all_gain_slider.grid(row=1, column=2, padx=5, pady=5, sticky='ew')

        # Display label for gain value
        self.all_gain_label = ttk.Label(table_frame, text="1.00x", width=6)
        self.all_gain_label.grid(row=1, column=3, padx=(2, 10), pady=5)

        def update_all_gain_label(*args):
            gain = self.bitplane_all_gain.get()
            self.all_gain_label.config(text=f"{gain:.2f}x")

        self.bitplane_all_gain.trace_add("write", update_all_gain_label)

        # Separator line after "All" row
        sep_frame = ttk.Frame(table_frame, height=2, relief='sunken')
        sep_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=(10, 10))

        # Create 8 rows for bit planes
        bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]
        self.gain_labels = []

        for i, label in enumerate(bit_labels):
            row = i + 3  # Start at row 3 (after header, All row, and separator)

            ttk.Label(table_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(table_frame, variable=self.bitplane_enable[i]).grid(row=row, column=1, padx=5, pady=5)

            def update_gain(slider_val, idx=i):
                gain = self._slider_to_gain(float(slider_val))
                self.bitplane_gain[idx].set(gain)

            gain_slider = ttk.Scale(table_frame, from_=-1, to=1, variable=self.bitplane_gain_slider[i], orient='horizontal',
                                    command=lambda v, idx=i: update_gain(v, idx))
            gain_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')

            # Gain value label
            gain_label = ttk.Label(table_frame, text="1.00x", width=6)
            gain_label.grid(row=row, column=3, padx=(2, 10), pady=5)
            self.gain_labels.append(gain_label)

            def update_gain_label(var_name, index, mode, idx=i):
                gain = self.bitplane_gain[idx].get()
                self.gain_labels[idx].config(text=f"{gain:.2f}x")

            self.bitplane_gain[i].trace_add("write", update_gain_label)

        table_frame.columnconfigure(2, weight=1)

        return self.control_panel

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

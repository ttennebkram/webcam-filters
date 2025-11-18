"""
Signals Ringing Effect

Advanced FFT-based frequency filter with interactive controls and visualization.
Supports multiple output modes including grayscale, RGB channels, and bit plane decomposition.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import os
import webbrowser
import math


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# FFT Filter Defaults
DEFAULT_FFT_RADIUS = 0  # Default radius for FFT low-frequency reject circle
DEFAULT_FFT_SMOOTHNESS = 0  # Default smoothness (0 = hard circle, 100 = very smooth transition)
DEFAULT_SHOW_FFT = False  # Default: don't show FFT visualization
DEFAULT_GAIN = 1  # Default gain (0.2 to 5, 1=no change)
DEFAULT_INVERT = False  # Default: don't invert
DEFAULT_OUTPUT_MODE = "grayscale_composite"  # Default output mode

# Butterworth Filter Algorithm Parameters
BUTTERWORTH_ORDER_MAX = 10.0  # Maximum Butterworth filter order
BUTTERWORTH_ORDER_MIN = 0.5  # Minimum Butterworth filter order
BUTTERWORTH_ORDER_RANGE = 9.5  # Order range (max - min)
BUTTERWORTH_SMOOTHNESS_SCALE = 100.0  # Smoothness parameter scale (0-100)
BUTTERWORTH_TARGET_ATTENUATION = 0.03  # Target filter attenuation at user radius (3%)
BUTTERWORTH_DIVISION_EPSILON = 1e-10  # Small epsilon to avoid division by zero

# Visualization Window Settings
VIZ_WINDOW_WIDTH = 600  # Visualization window width in pixels
VIZ_WINDOW_HEIGHT = 400  # Visualization window height in pixels
VIZ_FIGURE_WIDTH_INCHES = 6  # Matplotlib figure width in inches
VIZ_FIGURE_HEIGHT_INCHES = 4  # Matplotlib figure height in inches
VIZ_FIGURE_DPI = 100  # Figure dots per inch

# Visualization Graph Axis Limits
VIZ_Y_AXIS_MIN = -0.1  # Y-axis minimum (below 0 for clarity)
VIZ_Y_AXIS_MAX = 1.1  # Y-axis maximum (above 1 for clarity)
VIZ_X_AXIS_MIN = 0  # X-axis minimum
VIZ_X_AXIS_MAX = 400  # X-axis maximum (distance from FFT center in pixels)

# Visualization Reference Lines
VIZ_REF_LINE_ALPHA_MAJOR = 0.5  # Alpha for major reference lines (0, 1)
VIZ_REF_LINE_ALPHA_MINOR = 0.3  # Alpha for minor reference lines (0.03, 0.5)
VIZ_GRID_ALPHA = 0.3  # Alpha for background grid

# Visualization Line Styling - Bit Planes
VIZ_LINEWIDTH_BIT7_MSB = 2.0  # Bit 7 (MSB) line width
VIZ_LINEWIDTH_BIT6 = 1.7  # Bit 6 line width
VIZ_LINEWIDTH_BIT5 = 1.5  # Bit 5 line width
VIZ_LINEWIDTH_BIT4 = 1.3  # Bit 4 line width
VIZ_LINEWIDTH_VREF = 1.0  # Vertical reference line width

# Visualization Alpha Values - Bit Planes
VIZ_ALPHA_BIT7_MSB = 0.95  # Bit 7 (MSB) alpha
VIZ_ALPHA_BIT6 = 0.90  # Bit 6 alpha
VIZ_ALPHA_BIT5 = 0.85  # Bit 5 alpha
VIZ_ALPHA_BIT4 = 0.80  # Bit 4 alpha
VIZ_ALPHA_VREF = 0.4  # Vertical reference line alpha

# Visualization Font Sizes
VIZ_FONTSIZE_TITLE = 12  # Title font size
VIZ_FONTSIZE_AXIS_LABEL = 10  # Axis label font size
VIZ_FONTSIZE_LEGEND_BITPLANE = 7  # Bit plane mode legend font size
VIZ_FONTSIZE_LEGEND_GRAYSCALE = 8  # Grayscale mode legend font size
VIZ_FONTSIZE_LEGEND_RGB = 9  # RGB mode legend font size
VIZ_FONTSIZE_ANNOTATION = 9  # Annotation text font size

# Visualization Legend Settings
VIZ_LEGEND_NCOL_BITPLANE = 3  # Number of columns for bit plane legend
VIZ_LEGEND_NCOL_GRAYSCALE = 2  # Number of columns for grayscale legend

# FFT Visualization Settings
FFT_LOG_SCALE_MULTIPLIER = 20  # Multiplier for log magnitude spectrum (20*log)
FFT_CIRCLE_DASH_SEGMENTS = 60  # Number of segments for dashed circle rendering
FFT_CIRCLE_DASH_GAP_RATIO = 0.5  # Ratio for dash gaps (every other segment)
FFT_CIRCLE_OPACITY_BASE = 0.5  # Base opacity multiplier for filled circles
FFT_CIRCLE_THICKNESS = 2  # Line thickness for dotted circle borders


class SignalsRingingEffect(BaseUIEffect):
    """FFT-based high-pass filter with frequency domain masking and UI controls"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Variables for controls
        self.fft_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.fft_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)
        self.show_fft = tk.BooleanVar(value=DEFAULT_SHOW_FFT)
        self.gain = tk.DoubleVar(value=DEFAULT_GAIN)
        self.gain_display = tk.StringVar(value=f"{DEFAULT_GAIN:.1f}")
        self.invert = tk.BooleanVar(value=DEFAULT_INVERT)
        self.output_mode = tk.StringVar(value=DEFAULT_OUTPUT_MODE)

        # RGB channel controls
        self.red_enable = tk.BooleanVar(value=True)
        self.red_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.red_radius_slider = tk.IntVar(value=int(self._radius_to_slider(DEFAULT_FFT_RADIUS)))
        self.red_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        self.green_enable = tk.BooleanVar(value=True)
        self.green_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.green_radius_slider = tk.IntVar(value=int(self._radius_to_slider(DEFAULT_FFT_RADIUS)))
        self.green_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        self.blue_enable = tk.BooleanVar(value=True)
        self.blue_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.blue_radius_slider = tk.IntVar(value=int(self._radius_to_slider(DEFAULT_FFT_RADIUS)))
        self.blue_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        # Grayscale bit plane controls (8 bit planes: 7 MSB down to 0 LSB)
        self.bitplane_enable = []
        self.bitplane_radius = []
        self.bitplane_radius_slider = []
        self.bitplane_smoothness = []
        for i in range(8):
            self.bitplane_enable.append(tk.BooleanVar(value=True))
            self.bitplane_radius.append(tk.IntVar(value=0))  # Default to 0
            self.bitplane_radius_slider.append(tk.DoubleVar(value=0))  # Default to 0
            self.bitplane_smoothness.append(tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS))

        # "All" controls for grayscale bit planes
        self.bitplane_all_enable = tk.BooleanVar(value=True)
        self.bitplane_all_radius = tk.IntVar(value=0)
        self.bitplane_all_radius_slider = tk.DoubleVar(value=0)
        self.bitplane_all_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        # Color bit plane controls (3 colors × 8 bit planes each)
        self.color_bitplane_enable = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_radius = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_radius_slider = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_smoothness = {'red': [], 'green': [], 'blue': []}

        for color in ['red', 'green', 'blue']:
            for i in range(8):
                self.color_bitplane_enable[color].append(tk.BooleanVar(value=True))
                self.color_bitplane_radius[color].append(tk.IntVar(value=0))
                self.color_bitplane_radius_slider[color].append(tk.DoubleVar(value=0))
                self.color_bitplane_smoothness[color].append(tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS))

        # "All" controls for color bit planes
        self.color_bitplane_all_enable = {'red': tk.BooleanVar(value=True),
                                           'green': tk.BooleanVar(value=True),
                                           'blue': tk.BooleanVar(value=True)}
        self.color_bitplane_all_radius = {'red': tk.IntVar(value=0),
                                           'green': tk.IntVar(value=0),
                                           'blue': tk.IntVar(value=0)}
        self.color_bitplane_all_radius_slider = {'red': tk.DoubleVar(value=0),
                                                  'green': tk.DoubleVar(value=0),
                                                  'blue': tk.DoubleVar(value=0)}
        self.color_bitplane_all_smoothness = {'red': tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS),
                                               'green': tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS),
                                               'blue': tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)}

        # Visualization window
        self.viz_window = None
        self.viz_fig = None
        self.viz_ax = None
        self.viz_canvas = None

        # Difference window (shows what the filter removed)
        self.diff_window = None
        self.diff_label = None
        self.input_frame = None
        self.diff_frame = None

        # References for UI elements (will be set in _build_ui)
        self.rgb_table_frame = None
        self.rgb_expanded = None
        self.rgb_toggle_btn = None
        self.bitplane_table_frame = None
        self.bitplane_expanded = None
        self.bitplane_toggle_btn = None
        self.color_bitplane_notebook_frame = None
        self.color_bitplane_expanded = None
        self.color_bitplane_toggle_btn = None
        self.color_bp_selected_tab = None
        self.color_bp_tab_frames = {}
        self.color_bp_tab_buttons = {}
        self._scrollable_canvas = None

    @classmethod
    def get_name(cls):
        return "Signals Ringing"

    @classmethod
    def get_description(cls):
        return "FFT-based frequency filter with interactive controls and visualization"

    @classmethod
    def get_category(cls):
        return "signals"

    @classmethod
    def get_control_title(cls):
        return "FFT Filter"

    def _slider_to_radius(self, slider_value):
        """Convert linear slider value (0-100) to exponential radius (0-200+)

        Uses formula: radius = floor(e^(slider/25) - 1)
        This gives fine control at low values and larger steps at high values
        """
        return int(math.exp(slider_value / 25.0) - 1)

    def _radius_to_slider(self, radius):
        """Convert exponential radius value to linear slider position

        Inverse of _slider_to_radius: slider = 25 * ln(radius + 1)
        """
        if radius <= 0:
            return 0
        return 25.0 * math.log(radius + 1)

    def _open_url(self, url):
        """Open a URL in the default web browser"""
        webbrowser.open(url)

    def create_control_panel(self, parent):
        """Create Tkinter control panel for this effect"""
        self.control_panel = ttk.Frame(parent)
        self._build_ui()

        # Load saved settings BEFORE adding traces
        self._load_settings()

        # Auto-expand tables based on selected mode
        if self.output_mode.get() == "color_channels":
            self._toggle_rgb_table()
        if self.output_mode.get() == "grayscale_bitplanes":
            self._toggle_bitplane_table()
        if self.output_mode.get() == "color_bitplanes":
            self._toggle_color_bitplane_table()

        # Add traces to auto-select modes when controls change
        self.fft_radius.trace_add("write", self._on_grayscale_control_change)
        self.fft_smoothness.trace_add("write", self._on_grayscale_control_change)

        # RGB channel traces
        self.red_enable.trace_add("write", self._on_rgb_control_change)
        self.red_radius.trace_add("write", self._on_rgb_control_change)
        self.red_smoothness.trace_add("write", self._on_rgb_control_change)
        self.green_enable.trace_add("write", self._on_rgb_control_change)
        self.green_radius.trace_add("write", self._on_rgb_control_change)
        self.green_smoothness.trace_add("write", self._on_rgb_control_change)
        self.blue_enable.trace_add("write", self._on_rgb_control_change)
        self.blue_radius.trace_add("write", self._on_rgb_control_change)
        self.blue_smoothness.trace_add("write", self._on_rgb_control_change)

        # Bit plane traces
        for i in range(8):
            self.bitplane_enable[i].trace_add("write", self._on_bitplane_control_change)
            self.bitplane_radius[i].trace_add("write", self._on_bitplane_control_change)
            self.bitplane_smoothness[i].trace_add("write", self._on_bitplane_control_change)

        # Always create the visualization window
        self._create_visualization_window()

        # Update visualization with initial settings
        self._update_visualization()

        # Create the difference window
        self._create_diff_window()

        return self.control_panel

    def _build_ui(self):
        """Build the tkinter UI with scrollable canvas and all collapsible sections"""
        padding = {'padx': 10, 'pady': 3}

        # Create a canvas and scrollbar for scrolling (no border or highlight)
        canvas = tk.Canvas(self.control_panel, height=700, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.control_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Use scrollable_frame as container
        container = scrollable_frame

        # Add invisible spacer frame to maintain minimum width
        spacer = ttk.Frame(container, width=660, height=1)
        spacer.pack(fill='x')
        spacer.pack_propagate(False)

        # Store canvas reference for mousewheel scrolling
        self._scrollable_canvas = canvas

        def _on_mousewheel(event):
            # Platform-specific scrolling
            if event.num == 5 or event.delta < 0:
                self._scrollable_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                self._scrollable_canvas.yview_scroll(-1, "units")
            return "break"

        # Bind mousewheel scrolling
        if self.root_window:
            self.root_window.bind_all("<MouseWheel>", _on_mousewheel)
            self.root_window.bind_all("<Button-4>", _on_mousewheel)
            self.root_window.bind_all("<Button-5>", _on_mousewheel)

        # FFT Filter Section
        fft_frame = ttk.Frame(container, padding=0)
        fft_frame.pack(fill='x', padx=10, pady=0)

        # Show FFT checkbox (reduced padding)
        ttk.Checkbutton(fft_frame, text="Show FFT (instead of image)",
                       variable=self.show_fft,
                       command=self._on_show_fft_change).pack(anchor='w', pady=(5, 5))

        # Row 1: Grayscale Composite with its controls
        gs_composite_group = ttk.LabelFrame(fft_frame, text="", padding=(5, 0, 5, 2))
        gs_composite_group.pack(fill='x', pady=(5, 2))

        gs_columns = ttk.Frame(gs_composite_group)
        gs_columns.pack(fill='x')

        gs_left = ttk.Frame(gs_columns)
        gs_left.pack(side='left', anchor='n', padx=(0, 10))

        gs_right = ttk.Frame(gs_columns)
        gs_right.pack(side='left', fill='x', expand=True, anchor='n')

        # Radio button (reduced top padding)
        ttk.Radiobutton(gs_left, text="Grayscale Composite", value="grayscale_composite",
                       variable=self.output_mode).pack(anchor='nw', pady=(5, 0))

        # Controls for Grayscale Composite
        ttk.Label(gs_right, text="Filter Radius in pixels", wraplength=250).pack(anchor='w')

        # Second line with hyperlink
        second_line_container = ttk.Frame(gs_right)
        second_line_container.pack(anchor='w')
        ttk.Label(second_line_container, text='(controls the size of ').pack(side='left', padx=0)
        link_label = ttk.Label(second_line_container, text="FFT Ringing", foreground="blue", cursor="hand2")
        link_label.pack(side='left', padx=0)
        link_label.bind("<Button-1>", lambda e: self._open_url("https://en.wikipedia.org/wiki/Ringing_artifacts"))
        ttk.Label(second_line_container, text=')').pack(side='left', padx=0)

        radius_row = ttk.Frame(gs_right)
        radius_row.pack(fill='x')
        fft_radius_slider = ttk.Scale(radius_row, from_=5, to=200,
                                     variable=self.fft_radius, orient='horizontal',
                                     command=lambda v: self.fft_radius.set(int(float(v))))
        fft_radius_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(radius_row, textvariable=self.fft_radius, width=5).pack(side='left', padx=(5, 0))

        # Smoothness slider
        ttk.Label(gs_right, text="Filter Cutoff Smoothness 0-100 pixels", wraplength=250).pack(anchor='w', pady=(5, 0))
        ttk.Label(gs_right, text="(Butterworth offset, higher values reduce ringing)").pack(anchor='w')
        smooth_row = ttk.Frame(gs_right)
        smooth_row.pack(fill='x')
        fft_smoothness_slider = ttk.Scale(smooth_row, from_=0, to=100,
                                         variable=self.fft_smoothness, orient='horizontal',
                                         command=lambda v: self.fft_smoothness.set(int(float(v))))
        fft_smoothness_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(smooth_row, textvariable=self.fft_smoothness, width=5).pack(side='left', padx=(5, 0))

        # Row 2: Individual Color Channels - grouped section with table layout
        color_channels_group = ttk.LabelFrame(fft_frame, text="", padding=(5, 0, 5, 2))
        color_channels_group.pack(fill='x', pady=(0, 2))

        # Header with radio button and expand/collapse button
        rgb_header_frame = ttk.Frame(color_channels_group)
        rgb_header_frame.pack(fill='x', pady=(2, 2))

        ttk.Radiobutton(rgb_header_frame, text="Individual Color Channels", value="color_channels",
                       variable=self.output_mode, command=self._on_rgb_radio_select).pack(side='left')

        self.rgb_expanded = tk.BooleanVar(value=False)
        self.rgb_toggle_btn = ttk.Button(rgb_header_frame, text="▶", width=1,
                                        command=self._toggle_rgb_table)
        self.rgb_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(rgb_header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create table using grid layout (initially hidden)
        table_frame = ttk.Frame(color_channels_group)
        self.rgb_table_frame = table_frame

        # Header row
        ttk.Label(table_frame, text="").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Label(table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(table_frame, text="Smoothness (Butterworth offset)").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=5, padx=2, pady=2)

        # Red channel
        red_label = tk.Label(table_frame, text="Red", foreground="red", font=('TkDefaultFont', 9, 'bold'))
        red_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.red_enable).grid(row=1, column=1, padx=5, pady=5)

        def update_red_radius(slider_val):
            radius = self._slider_to_radius(float(slider_val))
            self.red_radius.set(radius)

        red_radius_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.red_radius_slider, orient='horizontal',
                                     command=lambda v: update_red_radius(v))
        red_radius_slider.grid(row=1, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.red_radius, width=4, foreground="red").grid(row=1, column=3, padx=(2, 10), pady=5)
        red_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.red_smoothness, orient='horizontal',
                                      command=lambda v: self.red_smoothness.set(int(float(v))))
        red_smooth_slider.grid(row=1, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.red_smoothness, width=4, foreground="red").grid(row=1, column=5, padx=2, pady=5)

        # Green channel
        green_label = tk.Label(table_frame, text="Green", foreground="green", font=('TkDefaultFont', 9, 'bold'))
        green_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.green_enable).grid(row=2, column=1, padx=5, pady=5)

        def update_green_radius(slider_val):
            radius = self._slider_to_radius(float(slider_val))
            self.green_radius.set(radius)

        green_radius_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.green_radius_slider, orient='horizontal',
                                       command=lambda v: update_green_radius(v))
        green_radius_slider.grid(row=2, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.green_radius, width=4, foreground="green").grid(row=2, column=3, padx=(2, 10), pady=5)
        green_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.green_smoothness, orient='horizontal',
                                        command=lambda v: self.green_smoothness.set(int(float(v))))
        green_smooth_slider.grid(row=2, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.green_smoothness, width=4, foreground="green").grid(row=2, column=5, padx=2, pady=5)

        # Blue channel
        blue_label = tk.Label(table_frame, text="Blue", foreground="blue", font=('TkDefaultFont', 9, 'bold'))
        blue_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.blue_enable).grid(row=3, column=1, padx=5, pady=5)

        def update_blue_radius(slider_val):
            radius = self._slider_to_radius(float(slider_val))
            self.blue_radius.set(radius)

        blue_radius_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.blue_radius_slider, orient='horizontal',
                                      command=lambda v: update_blue_radius(v))
        blue_radius_slider.grid(row=3, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.blue_radius, width=4, foreground="blue").grid(row=3, column=3, padx=(2, 10), pady=5)
        blue_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.blue_smoothness, orient='horizontal',
                                       command=lambda v: self.blue_smoothness.set(int(float(v))))
        blue_smooth_slider.grid(row=3, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.blue_smoothness, width=4, foreground="blue").grid(row=3, column=5, padx=2, pady=5)

        # Configure column weights
        table_frame.columnconfigure(2, weight=1)
        table_frame.columnconfigure(4, weight=1)

        # Row 3: Grayscale Bit Planes
        gs_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=(5, 0, 5, 2))
        gs_bitplanes_group.pack(fill='x', pady=(0, 2))

        header_frame = ttk.Frame(gs_bitplanes_group)
        header_frame.pack(fill='x', pady=(2, 2))

        bitplane_radio = ttk.Radiobutton(header_frame, text="Grayscale Bit Planes",
                                        value="grayscale_bitplanes",
                                        variable=self.output_mode,
                                        command=self._on_bitplane_radio_select)
        bitplane_radio.pack(side='left', anchor='w')

        self.bitplane_expanded = tk.BooleanVar(value=False)
        self.bitplane_toggle_btn = ttk.Button(header_frame, text="▶", width=1,
                                              command=self._toggle_bitplane_table)
        self.bitplane_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create table for bit planes (initially hidden)
        bitplane_table_frame = ttk.Frame(gs_bitplanes_group)
        self.bitplane_table_frame = bitplane_table_frame

        # Header row
        ttk.Label(bitplane_table_frame, text="Bit").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        ttk.Label(bitplane_table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(bitplane_table_frame, text="Smoothness (Butterworth offset)").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=5, padx=2, pady=2)

        # Create 8 rows for bit planes
        bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]
        for i, label in enumerate(bit_labels):
            row = i + 1

            ttk.Label(bitplane_table_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(bitplane_table_frame, variable=self.bitplane_enable[i]).grid(row=row, column=1, padx=5, pady=5)

            def update_radius(slider_val, idx=i):
                radius = self._slider_to_radius(float(slider_val))
                self.bitplane_radius[idx].set(radius)

            radius_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_radius_slider[i], orient='horizontal',
                                     command=lambda v, idx=i: update_radius(v, idx))
            radius_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')
            tk.Label(bitplane_table_frame, textvariable=self.bitplane_radius[i], width=4).grid(row=row, column=3, padx=(2, 10), pady=5)

            smooth_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_smoothness[i], orient='horizontal',
                                      command=lambda v, idx=i: self.bitplane_smoothness[idx].set(int(float(v))))
            smooth_slider.grid(row=row, column=4, padx=5, pady=5, sticky='ew')
            tk.Label(bitplane_table_frame, textvariable=self.bitplane_smoothness[i], width=4).grid(row=row, column=5, padx=2, pady=5)

        # "All" row at the bottom
        ttk.Label(bitplane_table_frame, text="All", font=('TkDefaultFont', 14, 'bold')).grid(row=9, column=0, padx=5, pady=5, sticky='e')

        def on_bitplane_all_enable_change(*args):
            enabled = self.bitplane_all_enable.get()
            for i in range(8):
                self.bitplane_enable[i].set(enabled)

        self.bitplane_all_enable.trace_add("write", on_bitplane_all_enable_change)
        ttk.Checkbutton(bitplane_table_frame, variable=self.bitplane_all_enable).grid(row=9, column=1, padx=5, pady=5)

        def update_all_bitplane_radius(slider_val):
            radius = self._slider_to_radius(float(slider_val))
            self.bitplane_all_radius.set(radius)
            for i in range(8):
                self.bitplane_radius[i].set(radius)
                self.bitplane_radius_slider[i].set(slider_val)

        all_radius_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_all_radius_slider, orient='horizontal',
                                     command=lambda v: update_all_bitplane_radius(v))
        all_radius_slider.grid(row=9, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(bitplane_table_frame, textvariable=self.bitplane_all_radius, width=4).grid(row=9, column=3, padx=(2, 10), pady=5)

        def update_all_bitplane_smoothness(val):
            smoothness = int(float(val))
            self.bitplane_all_smoothness.set(smoothness)
            for i in range(8):
                self.bitplane_smoothness[i].set(smoothness)

        all_smooth_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_all_smoothness, orient='horizontal',
                                      command=lambda v: update_all_bitplane_smoothness(v))
        all_smooth_slider.grid(row=9, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(bitplane_table_frame, textvariable=self.bitplane_all_smoothness, width=4).grid(row=9, column=5, padx=2, pady=5)

        bitplane_table_frame.columnconfigure(2, weight=1)
        bitplane_table_frame.columnconfigure(4, weight=1)

        # Row 4: Color Bit Planes
        color_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=(5, 0, 5, 2))
        color_bitplanes_group.pack(fill='x', pady=(0, 2))

        color_bp_header_frame = ttk.Frame(color_bitplanes_group)
        color_bp_header_frame.pack(fill='x', pady=(2, 2))

        color_bp_radio = ttk.Radiobutton(color_bp_header_frame, text="Color Bit Planes",
                                        value="color_bitplanes",
                                        variable=self.output_mode,
                                        command=self._on_color_bitplane_radio_select)
        color_bp_radio.pack(side='left', anchor='w')

        self.color_bitplane_expanded = tk.BooleanVar(value=False)
        self.color_bitplane_toggle_btn = ttk.Button(color_bp_header_frame, text="▶", width=1,
                                                    command=self._toggle_color_bitplane_table)
        self.color_bitplane_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(color_bp_header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create notebook with tabs for Red, Green, Blue
        color_bp_notebook_frame = ttk.Frame(color_bitplanes_group)
        self.color_bitplane_notebook_frame = color_bp_notebook_frame

        # Create custom tab bar
        tab_bar = ttk.Frame(color_bp_notebook_frame)
        tab_bar.pack(fill='x', pady=(0, 0))

        self.color_bp_selected_tab = tk.StringVar(value='red')

        tab_buttons_frame = tk.Frame(tab_bar, bg='gray85')
        tab_buttons_frame.pack(side='left', fill='x')

        tab_content_container = ttk.Frame(color_bp_notebook_frame, relief='solid', borderwidth=1)
        tab_content_container.pack(fill='both', expand=True)

        style = ttk.Style()

        # Create tabs for Red, Green, Blue
        for color_name, color_fg in [('Red', 'red'), ('Green', 'green'), ('Blue', 'DeepSkyBlue')]:
            color_key = color_name.lower()

            style_name = f"{color_key}_label.TLabel"
            style.configure(style_name, foreground=color_fg)

            # Create colored tab button
            tab_btn = tk.Label(tab_buttons_frame, text=color_name, foreground=color_fg,
                              relief='raised', borderwidth=1, padx=10, pady=5,
                              font=('TkDefaultFont', 9, 'bold'), bg='gray85')
            tab_btn.pack(side='left', padx=(0, 2))
            tab_btn.bind('<Button-1>', lambda e, c=color_key: self._switch_color_bp_tab(c))
            self.color_bp_tab_buttons[color_key] = tab_btn

            # Create frame for this tab's content
            tab_frame = ttk.Frame(tab_content_container)
            self.color_bp_tab_frames[color_key] = tab_frame

            # Create table for this color's bit planes
            table_frame = ttk.Frame(tab_frame)
            table_frame.pack(fill='both', expand=True, padx=5, pady=5)

            # Header row
            ttk.Label(table_frame, text="Bit", style=style_name).grid(row=0, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(table_frame, text="Enabled", style=style_name).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text="Filter Radius (pixels)", style=style_name).grid(row=0, column=2, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
            ttk.Label(table_frame, text="Smoothness (Butterworth offset)", style=style_name).grid(row=0, column=4, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=5, padx=2, pady=2)

            # Create 8 rows for bit planes
            bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]

            for i, label in enumerate(bit_labels):
                row = i + 1

                ttk.Label(table_frame, text=label, style=style_name).grid(row=row, column=0, padx=5, pady=5, sticky='e')
                ttk.Checkbutton(table_frame, variable=self.color_bitplane_enable[color_key][i]).grid(row=row, column=1, padx=5, pady=5)

                def update_color_bp_radius(slider_val, c=color_key, idx=i):
                    radius = self._slider_to_radius(float(slider_val))
                    self.color_bitplane_radius[c][idx].set(radius)

                radius_slider = ttk.Scale(table_frame, from_=0, to=100,
                                         variable=self.color_bitplane_radius_slider[color_key][i],
                                         orient='horizontal',
                                         command=lambda v, c=color_key, idx=i: update_color_bp_radius(v, c, idx))
                radius_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')

                tk.Label(table_frame, textvariable=self.color_bitplane_radius[color_key][i],
                        width=4, foreground=color_fg).grid(row=row, column=3, padx=(2, 10), pady=5)

                smooth_slider = ttk.Scale(table_frame, from_=0, to=100,
                                          variable=self.color_bitplane_smoothness[color_key][i],
                                          orient='horizontal',
                                          command=lambda v, c=color_key, idx=i: self.color_bitplane_smoothness[c][idx].set(int(float(v))))
                smooth_slider.grid(row=row, column=4, padx=5, pady=5, sticky='ew')

                tk.Label(table_frame, textvariable=self.color_bitplane_smoothness[color_key][i],
                        width=4, foreground=color_fg).grid(row=row, column=5, padx=2, pady=5)

            # "All" row at the bottom for this color
            tk.Label(table_frame, text="All", font=('TkDefaultFont', 14, 'bold'), foreground=color_fg).grid(row=9, column=0, padx=5, pady=5, sticky='e')

            def on_color_all_enable_change(c=color_key, *args):
                enabled = self.color_bitplane_all_enable[c].get()
                for i in range(8):
                    self.color_bitplane_enable[c][i].set(enabled)

            self.color_bitplane_all_enable[color_key].trace_add("write", on_color_all_enable_change)
            ttk.Checkbutton(table_frame, variable=self.color_bitplane_all_enable[color_key]).grid(row=9, column=1, padx=5, pady=5)

            def update_all_color_bp_radius(slider_val, c=color_key):
                radius = self._slider_to_radius(float(slider_val))
                self.color_bitplane_all_radius[c].set(radius)
                for i in range(8):
                    self.color_bitplane_radius[c][i].set(radius)
                    self.color_bitplane_radius_slider[c][i].set(slider_val)

            all_color_radius_slider = ttk.Scale(table_frame, from_=0, to=100,
                                               variable=self.color_bitplane_all_radius_slider[color_key],
                                               orient='horizontal',
                                               command=lambda v, c=color_key: update_all_color_bp_radius(v, c))
            all_color_radius_slider.grid(row=9, column=2, padx=5, pady=5, sticky='ew')
            tk.Label(table_frame, textvariable=self.color_bitplane_all_radius[color_key],
                    width=4, foreground=color_fg).grid(row=9, column=3, padx=(2, 10), pady=5)

            def update_all_color_bp_smoothness(val, c=color_key):
                smoothness = int(float(val))
                self.color_bitplane_all_smoothness[c].set(smoothness)
                for i in range(8):
                    self.color_bitplane_smoothness[c][i].set(smoothness)

            all_color_smooth_slider = ttk.Scale(table_frame, from_=0, to=100,
                                               variable=self.color_bitplane_all_smoothness[color_key],
                                               orient='horizontal',
                                               command=lambda v, c=color_key: update_all_color_bp_smoothness(v, c))
            all_color_smooth_slider.grid(row=9, column=4, padx=5, pady=5, sticky='ew')
            tk.Label(table_frame, textvariable=self.color_bitplane_all_smoothness[color_key],
                    width=4, foreground=color_fg).grid(row=9, column=5, padx=2, pady=5)

            table_frame.columnconfigure(2, weight=1)
            table_frame.columnconfigure(4, weight=1)

        # Show the red tab by default
        self._switch_color_bp_tab('red')

        # Common controls at bottom
        common_frame = ttk.Frame(fft_frame)
        common_frame.pack(fill='x', pady=(10, 0))

        # Note: Gain and Invert are now in the global controls panel

        # Separator
        ttk.Separator(container, orient='horizontal').pack(fill='x', pady=5)

        # Save Settings Button
        save_frame = ttk.Frame(container, padding=10)
        save_frame.pack(fill='x', **padding)
        ttk.Button(save_frame, text="Save Settings", command=self._save_settings).pack(fill='x')

    def _toggle_rgb_table(self):
        """Toggle the visibility of the RGB channels table"""
        if self.rgb_expanded.get():
            self.rgb_table_frame.pack_forget()
            self.rgb_toggle_btn.config(text="▶")
            self.rgb_expanded.set(False)
        else:
            self.rgb_table_frame.pack(fill='x', padx=10)
            self.rgb_toggle_btn.config(text="▼")
            self.rgb_expanded.set(True)
            self.output_mode.set("color_channels")

    def _on_rgb_radio_select(self):
        """Expand the RGB table when radio button is selected"""
        if not self.rgb_expanded.get():
            self._toggle_rgb_table()

    def _toggle_bitplane_table(self):
        """Toggle the visibility of the bit plane table"""
        if self.bitplane_expanded.get():
            self.bitplane_table_frame.pack_forget()
            self.bitplane_toggle_btn.config(text="▶")
            self.bitplane_expanded.set(False)
        else:
            self.bitplane_table_frame.pack(fill='x', padx=10)
            self.bitplane_toggle_btn.config(text="▼")
            self.bitplane_expanded.set(True)
            self.output_mode.set("grayscale_bitplanes")

    def _on_bitplane_radio_select(self):
        """Expand the bit plane table when radio button is selected"""
        if not self.bitplane_expanded.get():
            self._toggle_bitplane_table()

    def _toggle_color_bitplane_table(self):
        """Toggle the visibility of the color bit plane notebook"""
        if self.color_bitplane_expanded.get():
            self.color_bitplane_notebook_frame.pack_forget()
            self.color_bitplane_toggle_btn.config(text="▶")
            self.color_bitplane_expanded.set(False)
        else:
            self.color_bitplane_notebook_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.color_bitplane_toggle_btn.config(text="▼")
            self.color_bitplane_expanded.set(True)
            self.output_mode.set("color_bitplanes")

    def _on_color_bitplane_radio_select(self):
        """Expand the color bit plane table when radio button is selected"""
        if not self.color_bitplane_expanded.get():
            self._toggle_color_bitplane_table()

    def _switch_color_bp_tab(self, color_key):
        """Switch between Red, Green, Blue tabs in color bit plane section"""
        for key, btn in self.color_bp_tab_buttons.items():
            if key == color_key:
                btn.config(relief='sunken', bg='white')
            else:
                btn.config(relief='raised', bg='gray85')

        for frame in self.color_bp_tab_frames.values():
            frame.pack_forget()

        self.color_bp_tab_frames[color_key].pack(fill='both', expand=True, padx=5, pady=5)

        if self.root_window:
            self.root_window.update_idletasks()

        self.color_bp_selected_tab.set(color_key)

    def _on_rgb_control_change(self, *args):
        """Auto-select Individual Color Channels radio button when RGB controls are changed"""
        self.output_mode.set("color_channels")
        if not self.rgb_expanded.get():
            self._toggle_rgb_table()
        self._update_visualization()

    def _on_grayscale_control_change(self, *args):
        """Auto-select Grayscale Composite radio button when grayscale controls are changed"""
        self.output_mode.set("grayscale_composite")
        self._update_visualization()

    def _on_bitplane_control_change(self, *args):
        """Auto-select Grayscale Bit Planes radio button when bit plane controls are changed"""
        self.output_mode.set("grayscale_bitplanes")
        if not self.bitplane_expanded.get():
            self._toggle_bitplane_table()
        self._update_visualization()

    def _on_show_fft_change(self):
        """Handle show FFT checkbox change - no longer used as window is always shown"""
        pass

    def _create_visualization_window(self):
        """Create matplotlib window to visualize the filter curve"""
        if self.viz_window is not None or self.root_window is None:
            return

        self.viz_window = tk.Toplevel(self.root_window)
        self.viz_window.title("Filter Curve Visualization")
        self.viz_window.geometry(f"{VIZ_WINDOW_WIDTH}x{VIZ_WINDOW_HEIGHT}")

        self.viz_fig = Figure(figsize=(VIZ_FIGURE_WIDTH_INCHES, VIZ_FIGURE_HEIGHT_INCHES),
                             dpi=VIZ_FIGURE_DPI)
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_ax.set_xlabel('Distance from Center (pixels)')
        self.viz_ax.set_ylabel('Mask Value (0=blocked, 1=passed)')
        self.viz_ax.set_title('FFT Filter Transition Curve')
        self.viz_ax.grid(True, alpha=VIZ_GRID_ALPHA)
        self.viz_ax.set_ylim(VIZ_Y_AXIS_MIN, VIZ_Y_AXIS_MAX)

        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.viz_window)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.viz_window.protocol("WM_DELETE_WINDOW", self._close_visualization_window)

    def _close_visualization_window(self):
        """Close the visualization window"""
        if self.viz_window:
            self.viz_window.destroy()
            self.viz_window = None
            self.viz_fig = None
            self.viz_ax = None
            self.viz_canvas = None

    def _create_diff_window(self):
        """Create window to display difference between input and filtered output"""
        if self.diff_window is not None or self.root_window is None:
            print(f"DEBUG: Not creating diff window - diff_window={self.diff_window}, root_window={self.root_window}")
            return

        print("DEBUG: Creating difference window...")
        self.diff_window = tk.Toplevel(self.root_window)
        self.diff_window.title("Difference View (Conservation of Energy)")

        # Create label to hold the image
        self.diff_label = tk.Label(self.diff_window, bg='black')
        self.diff_label.pack()

        # Set geometry to match video dimensions
        self.diff_window.geometry(f"{self.width}x{self.height}")

        # Make sure window is visible
        self.diff_window.deiconify()
        self.diff_window.lift()
        self.diff_window.attributes('-topmost', True)
        self.diff_window.after(100, lambda: self.diff_window.attributes('-topmost', False))

        # Don't let closing this window close the app
        self.diff_window.protocol("WM_DELETE_WINDOW", self._close_diff_window)
        print(f"DEBUG: Difference window created: {self.diff_window}")

    def _close_diff_window(self):
        """Close the difference window"""
        if self.diff_window:
            self.diff_window.destroy()
            self.diff_window = None
            self.diff_label = None

    def _update_diff_window(self):
        """Update the difference window with the current diff frame"""
        if self.diff_window is None or self.diff_label is None or self.diff_frame is None:
            return

        # Convert BGR to RGB for display
        import PIL.Image
        import PIL.ImageTk
        diff_rgb = cv2.cvtColor(self.diff_frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(diff_rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=img)

        # Update label
        self.diff_label.imgtk = imgtk  # Keep reference to prevent garbage collection
        self.diff_label.configure(image=imgtk)

    def _update_visualization(self):
        """Update the visualization graph based on current mode and settings"""
        if not self.viz_ax or not self.viz_canvas:
            return

        # Clear previous plot
        self.viz_ax.clear()
        self.viz_ax.set_xlabel('Distance from Center (pixels)')
        self.viz_ax.set_ylabel('Mask Value (0=blocked, 1=passed)')
        self.viz_ax.set_title('FFT Filter Transition Curve')
        self.viz_ax.grid(True, alpha=VIZ_GRID_ALPHA)
        self.viz_ax.set_ylim(VIZ_Y_AXIS_MIN, VIZ_Y_AXIS_MAX)

        # Get current mode
        mode = self.output_mode.get()

        # Generate x-axis (distance from center)
        max_distance = 200
        distances = np.linspace(0, max_distance, 1000)

        if mode == "grayscale_composite":
            # Plot single curve for grayscale
            radius = self.fft_radius.get()
            smoothness = self.fft_smoothness.get()
            curve = self._compute_filter_curve(distances, radius, smoothness)
            self.viz_ax.plot(distances, curve, 'b-', linewidth=2, label='Grayscale')
            self.viz_ax.legend()

        elif mode == "color_channels":
            # Plot curves for each enabled RGB channel
            if self.red_enable.get():
                radius = self.red_radius.get()
                smoothness = self.red_smoothness.get()
                curve = self._compute_filter_curve(distances, radius, smoothness)
                self.viz_ax.plot(distances, curve, 'r-', linewidth=2, label='Red')

            if self.green_enable.get():
                radius = self.green_radius.get()
                smoothness = self.green_smoothness.get()
                curve = self._compute_filter_curve(distances, radius, smoothness)
                self.viz_ax.plot(distances, curve, 'g-', linewidth=2, label='Green')

            if self.blue_enable.get():
                radius = self.blue_radius.get()
                smoothness = self.blue_smoothness.get()
                curve = self._compute_filter_curve(distances, radius, smoothness)
                self.viz_ax.plot(distances, curve, 'b-', linewidth=2, label='Blue')

            self.viz_ax.legend()

        elif mode == "grayscale_bitplanes":
            # Plot curves for each enabled bit plane
            colors = plt.cm.viridis(np.linspace(0, 1, 8))
            for i in range(8):
                if self.bitplane_enable[i].get():
                    radius = self.bitplane_radius[i].get()
                    smoothness = self.bitplane_smoothness[i].get()
                    curve = self._compute_filter_curve(distances, radius, smoothness)
                    self.viz_ax.plot(distances, curve, color=colors[i], linewidth=1.5,
                                   label=f'Bit {7-i}', alpha=0.8)
            self.viz_ax.legend(fontsize=8, ncol=2)

        elif mode == "color_bitplanes":
            # Plot curves for the selected color channel's bit planes
            color_channel = self.color_bp_selected_tab.get()
            color_map = {'red': 'Reds', 'green': 'Greens', 'blue': 'Blues'}
            colors = plt.cm.get_cmap(color_map[color_channel])(np.linspace(0.3, 1, 8))

            for i in range(8):
                if self.color_bitplane_enable[color_channel][i].get():
                    radius = self.color_bitplane_radius[color_channel][i].get()
                    smoothness = self.color_bitplane_smoothness[color_channel][i].get()
                    curve = self._compute_filter_curve(distances, radius, smoothness)
                    self.viz_ax.plot(distances, curve, color=colors[i], linewidth=1.5,
                                   label=f'{color_channel.capitalize()} Bit {7-i}', alpha=0.8)
            self.viz_ax.legend(fontsize=8, ncol=2)

        # Redraw canvas
        self.viz_canvas.draw()

    def _compute_filter_curve(self, distances, radius, smoothness):
        """Compute the filter curve values for given distances"""
        if radius == 0:
            return np.ones_like(distances)

        if smoothness == 0:
            # Hard circle
            return np.where(distances <= radius, 0, 1)
        else:
            # Butterworth highpass filter
            order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE
            if order < BUTTERWORTH_ORDER_MIN:
                order = BUTTERWORTH_ORDER_MIN

            target_attenuation = BUTTERWORTH_TARGET_ATTENUATION
            shift_factor = np.power(1.0/target_attenuation - 1.0, 1.0 / (2.0 * order))
            effective_cutoff = radius * shift_factor

            with np.errstate(divide='ignore', invalid='ignore'):
                curve = 1.0 / (1.0 + np.power(effective_cutoff / (distances + BUTTERWORTH_DIVISION_EPSILON), 2 * order))
                curve = np.nan_to_num(curve, nan=0.0, posinf=0.0, neginf=0.0)
                curve = np.clip(curve, 0, 1)

            return curve

    def _save_settings(self):
        """Save all settings to a JSON file"""
        settings = {
            'fft_radius': self.fft_radius.get(),
            'fft_smoothness': self.fft_smoothness.get(),
            'show_fft': self.show_fft.get(),
            # Note: gain and invert are now in global settings
            'output_mode': self.output_mode.get(),
            # RGB channel settings
            'red_enable': self.red_enable.get(),
            'red_radius': self.red_radius.get(),
            'red_smoothness': self.red_smoothness.get(),
            'green_enable': self.green_enable.get(),
            'green_radius': self.green_radius.get(),
            'green_smoothness': self.green_smoothness.get(),
            'blue_enable': self.blue_enable.get(),
            'blue_radius': self.blue_radius.get(),
            'blue_smoothness': self.blue_smoothness.get(),
            # Grayscale bit plane settings
            'bitplane_enable': [bp.get() for bp in self.bitplane_enable],
            'bitplane_radius': [bp.get() for bp in self.bitplane_radius],
            'bitplane_smoothness': [bp.get() for bp in self.bitplane_smoothness],
            # Color bit plane settings
            'color_bitplane_enable': {
                'red': [bp.get() for bp in self.color_bitplane_enable['red']],
                'green': [bp.get() for bp in self.color_bitplane_enable['green']],
                'blue': [bp.get() for bp in self.color_bitplane_enable['blue']]
            },
            'color_bitplane_radius': {
                'red': [bp.get() for bp in self.color_bitplane_radius['red']],
                'green': [bp.get() for bp in self.color_bitplane_radius['green']],
                'blue': [bp.get() for bp in self.color_bitplane_radius['blue']]
            },
            'color_bitplane_smoothness': {
                'red': [bp.get() for bp in self.color_bitplane_smoothness['red']],
                'green': [bp.get() for bp in self.color_bitplane_smoothness['green']],
                'blue': [bp.get() for bp in self.color_bitplane_smoothness['blue']]
            },
            'color_bp_selected_tab': self.color_bp_selected_tab.get()
        }

        settings_file = os.path.expanduser('~/.signals_ringing_settings.json')
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _load_settings(self):
        """Load settings from JSON file if it exists"""
        settings_file = os.path.expanduser('~/.signals_ringing_settings.json')
        if not os.path.exists(settings_file):
            return

        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            self.fft_radius.set(settings.get('fft_radius', DEFAULT_FFT_RADIUS))
            self.fft_smoothness.set(settings.get('fft_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.show_fft.set(settings.get('show_fft', DEFAULT_SHOW_FFT))
            # Note: gain and invert are now loaded from global settings
            self.output_mode.set(settings.get('output_mode', DEFAULT_OUTPUT_MODE))

            # Load RGB channel settings
            self.red_enable.set(settings.get('red_enable', True))
            red_radius_val = settings.get('red_radius', DEFAULT_FFT_RADIUS)
            self.red_radius.set(red_radius_val)
            self.red_radius_slider.set(self._radius_to_slider(red_radius_val))
            self.red_smoothness.set(settings.get('red_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.green_enable.set(settings.get('green_enable', True))
            green_radius_val = settings.get('green_radius', DEFAULT_FFT_RADIUS)
            self.green_radius.set(green_radius_val)
            self.green_radius_slider.set(self._radius_to_slider(green_radius_val))
            self.green_smoothness.set(settings.get('green_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.blue_enable.set(settings.get('blue_enable', True))
            blue_radius_val = settings.get('blue_radius', DEFAULT_FFT_RADIUS)
            self.blue_radius.set(blue_radius_val)
            self.blue_radius_slider.set(self._radius_to_slider(blue_radius_val))
            self.blue_smoothness.set(settings.get('blue_smoothness', DEFAULT_FFT_SMOOTHNESS))

            # Load grayscale bit plane settings
            bitplane_enable = settings.get('bitplane_enable', [True] * 8)
            bitplane_radius = settings.get('bitplane_radius', [DEFAULT_FFT_RADIUS] * 8)
            bitplane_smoothness = settings.get('bitplane_smoothness', [DEFAULT_FFT_SMOOTHNESS] * 8)
            for i in range(8):
                self.bitplane_enable[i].set(bitplane_enable[i] if i < len(bitplane_enable) else True)
                radius_val = bitplane_radius[i] if i < len(bitplane_radius) else DEFAULT_FFT_RADIUS
                self.bitplane_radius[i].set(radius_val)
                self.bitplane_radius_slider[i].set(self._radius_to_slider(radius_val))
                self.bitplane_smoothness[i].set(bitplane_smoothness[i] if i < len(bitplane_smoothness) else DEFAULT_FFT_SMOOTHNESS)

            # Load color bit plane settings
            color_bitplane_enable = settings.get('color_bitplane_enable', {
                'red': [True] * 8,
                'green': [True] * 8,
                'blue': [True] * 8
            })
            color_bitplane_radius = settings.get('color_bitplane_radius', {
                'red': [DEFAULT_FFT_RADIUS] * 8,
                'green': [DEFAULT_FFT_RADIUS] * 8,
                'blue': [DEFAULT_FFT_RADIUS] * 8
            })
            color_bitplane_smoothness = settings.get('color_bitplane_smoothness', {
                'red': [DEFAULT_FFT_SMOOTHNESS] * 8,
                'green': [DEFAULT_FFT_SMOOTHNESS] * 8,
                'blue': [DEFAULT_FFT_SMOOTHNESS] * 8
            })

            for color in ['red', 'green', 'blue']:
                color_enable = color_bitplane_enable.get(color, [True] * 8)
                color_radius = color_bitplane_radius.get(color, [DEFAULT_FFT_RADIUS] * 8)
                color_smoothness = color_bitplane_smoothness.get(color, [DEFAULT_FFT_SMOOTHNESS] * 8)

                for i in range(8):
                    self.color_bitplane_enable[color][i].set(color_enable[i] if i < len(color_enable) else True)
                    radius_val = color_radius[i] if i < len(color_radius) else DEFAULT_FFT_RADIUS
                    self.color_bitplane_radius[color][i].set(radius_val)
                    self.color_bitplane_radius_slider[color][i].set(self._radius_to_slider(radius_val))
                    self.color_bitplane_smoothness[color][i].set(color_smoothness[i] if i < len(color_smoothness) else DEFAULT_FFT_SMOOTHNESS)

            # Restore selected color tab
            saved_tab = settings.get('color_bp_selected_tab', 'red')
            if saved_tab in ['red', 'green', 'blue']:
                self.color_bp_selected_tab.set(saved_tab)
                if self.output_mode.get() == "color_bitplanes":
                    self._switch_color_bp_tab(saved_tab)

        except Exception as e:
            print(f"Error loading settings: {e}")

    def _create_mask(self, distance, radius, smoothness, rows, cols):
        """Create high-pass mask with given parameters"""
        if radius == 0:
            return np.ones((rows, cols, 2), np.float32)

        if smoothness == 0:
            # Hard circle mask
            mask = np.ones((rows, cols, 2), np.float32)
            mask_area = distance <= radius
            mask[mask_area] = 0
        else:
            # Butterworth highpass filter
            order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE
            if order < BUTTERWORTH_ORDER_MIN:
                order = BUTTERWORTH_ORDER_MIN

            mask = np.ones((rows, cols, 2), np.float32)
            target_attenuation = BUTTERWORTH_TARGET_ATTENUATION
            shift_factor = np.power(1.0/target_attenuation - 1.0, 1.0 / (2.0 * order))
            effective_cutoff = radius * shift_factor

            with np.errstate(divide='ignore', invalid='ignore'):
                transition = 1.0 / (1.0 + np.power(effective_cutoff / (distance + BUTTERWORTH_DIVISION_EPSILON), 2 * order))
                transition = np.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
                transition = np.clip(transition, 0, 1)

            mask[:, :, 0] = transition
            mask[:, :, 1] = transition

        return mask

    def _apply_fft_to_channel(self, channel, radius, smoothness):
        """Apply FFT filter to a single channel"""
        # Compute FFT
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get dimensions
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Create mask
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = self._create_mask(distance, radius, smoothness, rows, cols)

        # Apply mask
        fshift = dft_shift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        high_pass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return high_pass

    def draw(self, frame, face_mask=None):
        """Apply FFT-based high-pass filter"""
        # Store input frame for difference calculation
        self.input_frame = frame.copy()

        output_mode = self.output_mode.get()

        if output_mode == "grayscale_composite":
            # Convert to grayscale and apply FFT
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = self._apply_fft_to_channel(gray, self.fft_radius.get(), self.fft_smoothness.get())

            # Note: Gain and invert are now applied globally in main.py

            # Convert back to BGR
            result = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        elif output_mode == "color_channels":
            # Split into BGR channels
            b, g, r = cv2.split(frame)

            # Apply FFT to each enabled channel
            if self.red_enable.get():
                r_filtered = self._apply_fft_to_channel(r, self.red_radius.get(), self.red_smoothness.get())
            else:
                r_filtered = np.zeros_like(r)

            if self.green_enable.get():
                g_filtered = self._apply_fft_to_channel(g, self.green_radius.get(), self.green_smoothness.get())
            else:
                g_filtered = np.zeros_like(g)

            if self.blue_enable.get():
                b_filtered = self._apply_fft_to_channel(b, self.blue_radius.get(), self.blue_smoothness.get())
            else:
                b_filtered = np.zeros_like(b)

            # Note: Gain and invert are now applied globally in main.py

            # Merge back
            result = cv2.merge([b_filtered, g_filtered, r_filtered])

        elif output_mode == "grayscale_bitplanes":
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Decompose into bit planes
            bit_planes = []
            for bit in range(8):
                bit_plane = ((gray >> bit) & 1) * 255
                bit_planes.append(bit_plane.astype(np.uint8))

            # Process each bit plane
            filtered_bit_planes = []
            for bit in range(8):
                ui_index = 7 - bit
                if self.bitplane_enable[ui_index].get():
                    filtered = self._apply_fft_to_channel(bit_planes[bit],
                                                          self.bitplane_radius[ui_index].get(),
                                                          self.bitplane_smoothness[ui_index].get())
                    filtered_bit_planes.append(filtered)
                else:
                    filtered_bit_planes.append(bit_planes[bit])

            # Reconstruct grayscale image
            num_enabled = sum(1 for i in range(8) if self.bitplane_enable[i].get())

            if num_enabled == 0:
                result = np.zeros_like(gray, dtype=np.uint8)
            elif num_enabled == 1:
                for ui_index in range(8):
                    if self.bitplane_enable[ui_index].get():
                        bit = 7 - ui_index
                        binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                        result = binary_plane * 255
                        break
            else:
                reconstructed = np.zeros_like(gray, dtype=np.uint8)
                for bit in range(8):
                    ui_index = 7 - bit
                    if self.bitplane_enable[ui_index].get():
                        binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                        reconstructed += (binary_plane << bit)
                result = reconstructed

            # Note: Gain and invert are now applied globally in main.py

            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        else:  # color_bitplanes
            # Split into BGR channels
            b, g, r = cv2.split(frame)
            channels = {'blue': b, 'green': g, 'red': r}
            filtered_channels = {}

            for color_name, channel in channels.items():
                # Decompose into bit planes
                bit_planes = []
                for bit in range(8):
                    bit_plane = ((channel >> bit) & 1) * 255
                    bit_planes.append(bit_plane.astype(np.uint8))

                # Process each bit plane
                filtered_bit_planes = []
                for bit in range(8):
                    ui_index = 7 - bit
                    if self.color_bitplane_enable[color_name][ui_index].get():
                        filtered = self._apply_fft_to_channel(bit_planes[bit],
                                                              self.color_bitplane_radius[color_name][ui_index].get(),
                                                              self.color_bitplane_smoothness[color_name][ui_index].get())
                        filtered_bit_planes.append(filtered)
                    else:
                        filtered_bit_planes.append(bit_planes[bit])

                # Reconstruct channel
                num_enabled = sum(1 for i in range(8) if self.color_bitplane_enable[color_name][i].get())

                if num_enabled == 0:
                    filtered_channels[color_name] = np.zeros_like(channel, dtype=np.uint8)
                elif num_enabled == 1:
                    for ui_index in range(8):
                        if self.color_bitplane_enable[color_name][ui_index].get():
                            bit = 7 - ui_index
                            binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                            filtered_channels[color_name] = binary_plane * 255
                            break
                else:
                    reconstructed = np.zeros_like(channel, dtype=np.uint8)
                    for bit in range(8):
                        ui_index = 7 - bit
                        if self.color_bitplane_enable[color_name][ui_index].get():
                            binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                            reconstructed += (binary_plane << bit)
                    filtered_channels[color_name] = reconstructed

            # Note: Gain and invert are now applied globally in main.py

            # Merge channels (BGR order)
            result = cv2.merge([filtered_channels['blue'], filtered_channels['green'], filtered_channels['red']])

        # Calculate absolute difference between input and filtered output
        self.diff_frame = cv2.absdiff(self.input_frame, result)

        # Update difference window display if it exists
        self._update_diff_window()

        return result

    def cleanup(self):
        """Clean up resources"""
        self._close_visualization_window()
        self._close_diff_window()

#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
import os
import signal
import tkinter as tk
from tkinter import ttk
import threading
import argparse
import json


# Configuration constants - FFT Filter
DEFAULT_FFT_RADIUS = 30  # Default radius for FFT low-frequency reject circle
DEFAULT_FFT_SMOOTHNESS = 0  # Default smoothness (0 = hard circle, 100 = very smooth transition)
DEFAULT_SHOW_FFT = False  # Default: don't show FFT visualization
DEFAULT_GAIN = 0  # Default gain (-5 to +5)
DEFAULT_INVERT = False  # Default: don't invert
DEFAULT_OUTPUT_MODE = "grayscale_composite"  # Default output mode


class FFTFilter:
    """FFT-based high-pass filter with frequency domain masking"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.radius = DEFAULT_FFT_RADIUS  # Radius of low-frequency reject circle
        self.smoothness = DEFAULT_FFT_SMOOTHNESS  # Smoothness of transition (0-100)
        self.show_fft = DEFAULT_SHOW_FFT  # Whether to show FFT visualization

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply FFT-based high-pass filter"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute FFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get dimensions
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create high-pass mask (reject low frequencies in center)
        center_y, center_x = crow, ccol
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        if self.smoothness == 0:
            # Hard circle mask
            mask = np.ones((rows, cols, 2), np.float32)
            mask_area = distance <= self.radius
            mask[mask_area] = 0
        else:
            # Smooth transition using sigmoid, but clamped to preserve hard cutoff inside radius
            # Smoothness controls the width of the transition zone
            sigma = self.smoothness / 10.0  # Scale smoothness to reasonable sigma range
            mask = np.ones((rows, cols, 2), np.float32)
            # Create smooth transition from 0 at center to 1 at radius
            # Sigmoid centered at radius
            transition = 1.0 / (1.0 + np.exp(-(distance - self.radius) / sigma))
            # Clamp to ensure full blocking inside radius - no sigmoid creep!
            # When distance < radius, force to 0 to preserve maximum ringing
            transition = np.clip(transition, 0, 1)
            transition[distance < self.radius] = 0
            mask[:, :, 0] = transition
            mask[:, :, 1] = transition

        # Apply mask to FFT
        fshift = dft_shift * mask

        # If showing FFT visualization
        if self.show_fft:
            # Compute magnitude spectrum for visualization
            magnitude_spectrum = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)

            # Normalize to 0-255
            magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Convert to 3-channel for drawing circle
            fft_display = cv2.cvtColor(magnitude_normalized, cv2.COLOR_GRAY2BGR)

            # Calculate opacity based on smoothness
            # smoothness 0 = 50% opacity, smoothness 100 = 0% opacity (25% at smoothness 50)
            circle_opacity = 0.5 * (1.0 - self.smoothness / 100.0)

            # Draw red circle with variable opacity showing the reject region
            overlay = fft_display.copy()
            cv2.circle(overlay, (ccol, crow), self.radius, (0, 0, 255), -1)
            fft_display = cv2.addWeighted(fft_display, 1.0 - circle_opacity, overlay, circle_opacity, 0)

            return fft_display

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        high_pass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return high_pass


class ControlPanel:
    """Tkinter GUI control panel for filter parameters"""

    def __init__(self, width, height, available_cameras, selected_camera_id):
        self.width = width
        self.height = height
        self.available_cameras = available_cameras
        self.camera_changed = False
        self.new_camera_id = None

        # Create the main window
        self.root = tk.Tk()
        self.root.title("FFT Ring Settings")
        # Don't set geometry yet - will auto-size after building UI

        # Camera selection variable
        self.selected_camera = tk.IntVar(value=selected_camera_id)

        # Effect toggle (shared with main loop) - inverted logic: show_original = effect disabled
        self.show_original = tk.BooleanVar(value=False)

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
        self.red_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        self.green_enable = tk.BooleanVar(value=True)
        self.green_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.green_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        self.blue_enable = tk.BooleanVar(value=True)
        self.blue_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.blue_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)

        self._build_ui()

        # Load saved settings if they exist
        self._load_settings()

        # Auto-size window to fit all content
        self.root.update_idletasks()  # Ensure all widgets are laid out
        width = 650  # Fixed width (wider for table layout)
        # Get the required height from the container
        height = self.root.winfo_reqheight()
        # Add a small buffer
        height = min(height + 20, 1000)  # Cap at 1000px
        self.root.geometry(f"{width}x{height}")

        # Flag to check if window is still open
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Make sure window is visible and brought to front
        self.root.deiconify()
        self.root.lift()

    def _build_ui(self):
        """Build the tkinter UI"""
        padding = {'padx': 10, 'pady': 3}
        container = self.root

        # Camera Selection Section
        camera_frame = ttk.LabelFrame(container, text="Camera Selection", padding=10)
        camera_frame.pack(fill='x', **padding)

        ttk.Label(camera_frame, text="Select Camera:").pack(anchor='w')

        # Create radio buttons for each camera
        for cam in self.available_cameras:
            cam_text = f"Camera {cam['id']}: {cam['width']}x{cam['height']}"
            ttk.Radiobutton(camera_frame, text=cam_text, value=cam['id'],
                           variable=self.selected_camera,
                           command=self._on_camera_change).pack(anchor='w', pady=2)

        # Source Section
        source_frame = ttk.LabelFrame(container, text="Source", padding=10)
        source_frame.pack(fill='x', **padding)

        ttk.Checkbutton(source_frame, text="Show Original Image (disable all effects)",
                       variable=self.show_original).pack(anchor='w')

        # FFT Filter Section
        fft_frame = ttk.LabelFrame(container, text="FFT Filter", padding=10)
        fft_frame.pack(fill='x', **padding)

        # Create two-column layout
        columns_frame = ttk.Frame(fft_frame)
        columns_frame.pack(fill='x')

        # Left column - Output Mode Radio Buttons
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side='left', anchor='n', padx=(0, 10))

        # Right column - Controls for each mode
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side='left', fill='x', expand=True, anchor='n')

        # Header
        ttk.Label(left_column, text="Output Mode:").pack(anchor='w', pady=(0, 5))

        # Row 1: Grayscale Composite with its controls - grouped in a frame
        gs_composite_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        gs_composite_group.pack(fill='x', pady=(5, 10))

        # Create inner two-column layout for this group
        gs_columns = ttk.Frame(gs_composite_group)
        gs_columns.pack(fill='x')

        gs_left = ttk.Frame(gs_columns)
        gs_left.pack(side='left', anchor='n', padx=(0, 10))

        gs_right = ttk.Frame(gs_columns)
        gs_right.pack(side='left', fill='x', expand=True, anchor='n')

        # Radio button
        ttk.Radiobutton(gs_left, text="Grayscale Composite", value="grayscale_composite",
                       variable=self.output_mode).pack(anchor='nw', pady=(20, 0))

        # Controls for Grayscale Composite
        # Radius slider
        ttk.Label(gs_right, text="Filter Radius in pixels", wraplength=250).pack(anchor='w')
        radius_row = ttk.Frame(gs_right)
        radius_row.pack(fill='x')
        fft_radius_slider = ttk.Scale(radius_row, from_=5, to=200,
                                     variable=self.fft_radius, orient='horizontal',
                                     command=lambda v: self.fft_radius.set(int(float(v))))
        fft_radius_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(radius_row, textvariable=self.fft_radius, width=5).pack(side='left', padx=(5, 0))

        # Smoothness slider
        ttk.Label(gs_right, text="Filter Cutoff Smoothness (0-100 pixels)", wraplength=250).pack(anchor='w', pady=(5, 0))
        smooth_row = ttk.Frame(gs_right)
        smooth_row.pack(fill='x')
        fft_smoothness_slider = ttk.Scale(smooth_row, from_=0, to=100,
                                         variable=self.fft_smoothness, orient='horizontal',
                                         command=lambda v: self.fft_smoothness.set(int(float(v))))
        fft_smoothness_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(smooth_row, textvariable=self.fft_smoothness, width=5).pack(side='left', padx=(5, 0))

        # Row 2: Grayscale Bit Planes - grouped section
        gs_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        gs_bitplanes_group.pack(fill='x', pady=(0, 10))

        ttk.Radiobutton(gs_bitplanes_group, text="Grayscale Bit Planes", value="grayscale_bitplanes",
                       variable=self.output_mode).pack(anchor='w')

        # Row 3: Individual Color Channels - grouped section with table layout
        color_channels_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        color_channels_group.pack(fill='x', pady=(0, 10))

        ttk.Radiobutton(color_channels_group, text="Individual Color Channels", value="color_channels",
                       variable=self.output_mode).pack(anchor='w', pady=(0, 5))

        # Create table using grid layout
        table_frame = ttk.Frame(color_channels_group)
        table_frame.pack(fill='x', pady=(5, 0))

        # Header row (row 0)
        ttk.Label(table_frame, text="").grid(row=0, column=0, padx=5, pady=2, sticky='w')  # Empty cell for color labels
        ttk.Label(table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)  # Value label column
        ttk.Label(table_frame, text="Smoothness (Sigmoid/10)").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=5, padx=2, pady=2)  # Value label column

        # Red channel (row 1)
        red_label = tk.Label(table_frame, text="Red", foreground="red", font=('TkDefaultFont', 9, 'bold'))
        red_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.red_enable).grid(row=1, column=1, padx=5, pady=5)
        red_radius_slider = ttk.Scale(table_frame, from_=5, to=200, variable=self.red_radius, orient='horizontal',
                                     command=lambda v: self.red_radius.set(int(float(v))))
        red_radius_slider.grid(row=1, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.red_radius, width=4, foreground="red").grid(row=1, column=3, padx=(2, 10), pady=5)
        red_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.red_smoothness, orient='horizontal',
                                      command=lambda v: self.red_smoothness.set(int(float(v))))
        red_smooth_slider.grid(row=1, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.red_smoothness, width=4, foreground="red").grid(row=1, column=5, padx=2, pady=5)

        # Green channel (row 2)
        green_label = tk.Label(table_frame, text="Green", foreground="green", font=('TkDefaultFont', 9, 'bold'))
        green_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.green_enable).grid(row=2, column=1, padx=5, pady=5)
        green_radius_slider = ttk.Scale(table_frame, from_=5, to=200, variable=self.green_radius, orient='horizontal',
                                       command=lambda v: self.green_radius.set(int(float(v))))
        green_radius_slider.grid(row=2, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.green_radius, width=4, foreground="green").grid(row=2, column=3, padx=(2, 10), pady=5)
        green_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.green_smoothness, orient='horizontal',
                                        command=lambda v: self.green_smoothness.set(int(float(v))))
        green_smooth_slider.grid(row=2, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.green_smoothness, width=4, foreground="green").grid(row=2, column=5, padx=2, pady=5)

        # Blue channel (row 3)
        blue_label = tk.Label(table_frame, text="Blue", foreground="blue", font=('TkDefaultFont', 9, 'bold'))
        blue_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        ttk.Checkbutton(table_frame, variable=self.blue_enable).grid(row=3, column=1, padx=5, pady=5)
        blue_radius_slider = ttk.Scale(table_frame, from_=5, to=200, variable=self.blue_radius, orient='horizontal',
                                      command=lambda v: self.blue_radius.set(int(float(v))))
        blue_radius_slider.grid(row=3, column=2, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.blue_radius, width=4, foreground="blue").grid(row=3, column=3, padx=(2, 10), pady=5)
        blue_smooth_slider = ttk.Scale(table_frame, from_=0, to=100, variable=self.blue_smoothness, orient='horizontal',
                                       command=lambda v: self.blue_smoothness.set(int(float(v))))
        blue_smooth_slider.grid(row=3, column=4, padx=5, pady=5, sticky='ew')
        tk.Label(table_frame, textvariable=self.blue_smoothness, width=4, foreground="blue").grid(row=3, column=5, padx=2, pady=5)

        # Configure column weights for proper expansion
        table_frame.columnconfigure(2, weight=1)  # Filter Radius column expands
        table_frame.columnconfigure(4, weight=1)  # Smoothness column expands

        # Row 4: Color Bitplanes - grouped section
        color_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        color_bitplanes_group.pack(fill='x', pady=(0, 10))

        ttk.Radiobutton(color_bitplanes_group, text="Color Bitplanes", value="color_bitplanes",
                       variable=self.output_mode).pack(anchor='w')

        # Common controls at bottom
        common_frame = ttk.Frame(fft_frame)
        common_frame.pack(fill='x', pady=(10, 0))

        # Show FFT checkbox
        ttk.Checkbutton(common_frame, text="Show FFT (instead of image)", variable=self.show_fft).pack(anchor='w')

        # Gain slider with value on right
        ttk.Label(common_frame, text="Gain (-5 to +5)").pack(anchor='w', pady=(5, 0))
        gain_row = ttk.Frame(common_frame)
        gain_row.pack(fill='x')
        gain_slider = ttk.Scale(gain_row, from_=-5, to=5,
                               variable=self.gain, orient='horizontal',
                               command=lambda v: self.gain_display.set(f"{float(v):.1f}"))
        gain_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(gain_row, textvariable=self.gain_display, width=5).pack(side='left', padx=(5, 0))

        # Invert checkbox
        ttk.Checkbutton(common_frame, text="Invert", variable=self.invert).pack(anchor='w', pady=(5, 0))

        # Separator
        ttk.Separator(container, orient='horizontal').pack(fill='x', pady=5)

        # Save Settings Button
        save_frame = ttk.Frame(container, padding=10)
        save_frame.pack(fill='x', **padding)
        ttk.Button(save_frame, text="Save Settings", command=self._save_settings).pack(fill='x')

    def _save_settings(self):
        """Save all settings to a JSON file"""
        settings = {
            'fft_radius': self.fft_radius.get(),
            'fft_smoothness': self.fft_smoothness.get(),
            'show_fft': self.show_fft.get(),
            'show_original': self.show_original.get(),
            'gain': self.gain.get(),
            'invert': self.invert.get(),
            'output_mode': self.output_mode.get()
        }

        settings_file = os.path.expanduser('~/.webcam_filter_settings.json')
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"Settings saved to {settings_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _load_settings(self):
        """Load settings from JSON file if it exists"""
        settings_file = os.path.expanduser('~/.webcam_filter_settings.json')
        if not os.path.exists(settings_file):
            return

        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            # Apply loaded settings
            self.fft_radius.set(settings.get('fft_radius', DEFAULT_FFT_RADIUS))
            self.fft_smoothness.set(settings.get('fft_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.show_fft.set(settings.get('show_fft', DEFAULT_SHOW_FFT))
            self.show_original.set(settings.get('show_original', False))
            self.gain.set(settings.get('gain', DEFAULT_GAIN))
            self.invert.set(settings.get('invert', DEFAULT_INVERT))
            self.output_mode.set(settings.get('output_mode', DEFAULT_OUTPUT_MODE))

            print(f"Settings loaded from {settings_file}")
        except Exception as e:
            print(f"Error loading settings: {e}")

    def _on_camera_change(self):
        """Handle camera selection change"""
        self.camera_changed = True
        self.new_camera_id = self.selected_camera.get()

    def _on_close(self):
        """Handle window close"""
        self.running = False
        self.root.quit()

    def update(self):
        """Update the tkinter event loop - call this periodically"""
        if self.running:
            try:
                self.root.update()
            except tk.TclError:
                self.running = False


class CombinedSketchFilter:
    """FFT-based filter for webcam"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.fft = FFTFilter(width, height)

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply FFT filter and return result"""
        # If showing FFT visualization, return that directly
        if self.fft.show_fft:
            return self.fft.draw(frame)

        # Apply FFT filter
        fft_result = self.fft.draw(frame)

        # Convert to 3-channel for display
        result_3channel = cv2.cvtColor(fft_result, cv2.COLOR_GRAY2BGR)

        return result_3channel


class EdgeDetector:
    """Detect edges and surfaces using computer vision"""

    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

    def detect(self, frame):
        """Detect face and edges, return obstacle mask"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Detect edges using Canny - lower thresholds for more edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)  # Lower thresholds = more edges

        # Dilate edges more to make them much more prominent
        kernel = np.ones((7, 7), np.uint8)  # Bigger kernel
        edges = cv2.dilate(edges, kernel, iterations=2)  # More iterations

        # Add edges to mask
        mask = cv2.bitwise_or(mask, edges)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Expand the face region a bit for better flow effect
                expand = 20
                x = max(0, x - expand)
                y = max(0, y - expand)
                w = min(width - x, w + 2 * expand)
                h = min(height - y, h + 2 * expand)

                # Create elliptical mask for more natural flow
                center = (x + w // 2, y + h // 2)
                axes = (w // 2, h // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        return mask


def signal_handler(sig, frame):
    """Handle Ctrl+C - force exit immediately"""
    print("\n\nCtrl+C detected - exiting NOW...")
    cv2.destroyAllWindows()
    os._exit(0)


def find_available_cameras():
    """Find all available cameras"""
    available_cameras = []

    print("Scanning for available cameras...")
    for camera_id in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append({
                    'id': camera_id,
                    'width': width,
                    'height': height
                })
                print(f"  Found camera {camera_id}: {width}x{height}")
            cap.release()

    return available_cameras


def main():
    """Main application loop"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='FFT Filter for Webcam')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera ID to use (default: highest numbered camera)')
    args = parser.parse_args()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Find available cameras
    available_cameras = find_available_cameras()

    if not available_cameras:
        print("\nError: No webcams found!")
        print("Please check that:")
        print("  1. Your webcam is connected")
        print("  2. No other application is using the webcam")
        print("  3. You have granted camera permissions")
        return

    # Select camera based on command line argument or use highest numbered camera
    if args.camera is not None:
        # Find camera with specified ID
        selected_camera = None
        for cam in available_cameras:
            if cam['id'] == args.camera:
                selected_camera = cam
                break

        if selected_camera is None:
            print(f"\nError: Camera {args.camera} not found!")
            print("Available cameras:")
            for cam in available_cameras:
                print(f"  Camera {cam['id']}: {cam['width']}x{cam['height']}")
            return

        print(f"Using camera {selected_camera['id']} (from command line)")
    else:
        # Use highest numbered camera
        selected_camera = max(available_cameras, key=lambda c: c['id'])
        print(f"Using camera {selected_camera['id']} (highest numbered camera)")
        print(f"To use a different camera, run with: --camera <id>")

    # Initialize selected webcam
    print(f"\nInitializing camera {selected_camera['id']}...")
    cap = cv2.VideoCapture(selected_camera['id'], cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open selected webcam")
        return

    # Get webcam dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Webcam initialized: {width}x{height}")
    print("Controls:")
    print("  SPACEBAR - Toggle effect on/off")
    print("  Q, ESC, or Ctrl+C - Quit")
    print("  Use control panel to adjust parameters")

    # Start window thread for better event handling and native controls
    cv2.startWindowThread()

    # Initialize combined sketch filter
    sketch = CombinedSketchFilter(width, height)

    # Create tkinter control panel
    controls = ControlPanel(width, height, available_cameras, selected_camera['id'])

    # Create main window for video display
    window_name = 'Sketch Filter'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size
    cv2.resizeWindow(window_name, width, height)
    # Move window to avoid overlapping with control panel (offset by 420 pixels to the right)
    cv2.moveWindow(window_name, 420, 0)

    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame")
            print("Retrying...")
            time.sleep(0.1)
            continue

        # Update tkinter control panel
        controls.update()

        # Check if control panel was closed
        if not controls.running:
            break

        # Check if camera was changed
        if controls.camera_changed:
            controls.camera_changed = False
            new_camera_id = controls.new_camera_id
            print(f"\nSwitching to camera {new_camera_id}...")

            # Release current camera
            cap.release()

            # Open new camera
            cap = cv2.VideoCapture(new_camera_id, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                print(f"Error: Could not open camera {new_camera_id}")
                # Try to reopen previous camera
                cap = cv2.VideoCapture(selected_camera['id'], cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    print(f"Reverted to camera {selected_camera['id']}")
                    controls.selected_camera.set(selected_camera['id'])
                else:
                    print("Fatal error: Could not reopen any camera")
                    break
            else:
                # Update selected camera
                for cam in available_cameras:
                    if cam['id'] == new_camera_id:
                        selected_camera = cam
                        break

                # Get new camera dimensions
                new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Reinitialize sketch filter with new dimensions
                sketch = CombinedSketchFilter(new_width, new_height)

                print(f"Now using camera {selected_camera['id']} at {new_width}x{new_height}")

                # Skip this frame and get a fresh one from the new camera
                continue

        # Mirror the image (flip horizontally)
        frame = cv2.flip(frame, 1)

        # Read values from tkinter controls and update parameters
        sketch.fft.radius = controls.fft_radius.get()
        sketch.fft.smoothness = controls.fft_smoothness.get()
        sketch.fft.show_fft = controls.show_fft.get()

        # Get effect enabled state from control panel (inverted from show_original)
        effect_enabled = not controls.show_original.get()

        if effect_enabled:
            # Sketch filter mode
            sketch.update()
            result = sketch.draw(frame)

            # Apply gain (multiply by 2^gain)
            gain = controls.gain.get()
            if gain != 0:
                result = np.clip(result * (2 ** gain), 0, 255).astype(np.uint8)

            # Apply invert
            if controls.invert.get():
                result = 255 - result

            # Effect enabled
        else:
            # Preview mode: show raw webcam
            result = frame.copy()
            # Effect disabled

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        # Display FPS and parameters at top left
        y_offset = 30
        line_height = 30
        cv2.putText(result, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        cv2.putText(result, f"FFT Radius: {sketch.fft.radius}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"FFT Smoothness: {sketch.fft.smoothness}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Show FFT: {sketch.fft.show_fft}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        # Display effect status
        status_text = "EFFECT ON" if effect_enabled else "EFFECT OFF (SPACEBAR to toggle)"
        status_color = (0, 255, 0) if effect_enabled else (0, 165, 255)  # Green if on, orange if off
        cv2.putText(result, status_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Check if frame is almost black (might be wrong camera)
        mean_brightness = np.mean(frame)
        if mean_brightness < 10:  # Very dark frame
            # Display warning in red at center
            h, w = result.shape[:2]
            warning_text = "ALMOST BLACK SCREEN!"
            camera_text = f"Try: --camera <id> (currently using camera {selected_camera['id']})"

            # Calculate text size for centering
            (w1, h1), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            (w2, h2), _ = cv2.getTextSize(camera_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Draw warnings in red
            cv2.putText(result, warning_text, (w//2 - w1//2, h//2 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(result, camera_text, (w//2 - w2//2, h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the result
        cv2.imshow(window_name, result)

        # Check if video window was closed via close button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Handle keyboard input (wrapped in try for Ctrl+C handling)
        try:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or Esc key
                print("\nExiting...")
                break
            # Note: Spacebar toggle is handled by tkinter binding in ControlPanel
        except KeyboardInterrupt:
            print("\nCtrl+C in loop - force exiting...")
            cv2.destroyAllWindows()
            os._exit(0)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C detected - force exiting...")
        cv2.destroyAllWindows()
        os._exit(0)  # Force immediate exit
    finally:
        cv2.destroyAllWindows()

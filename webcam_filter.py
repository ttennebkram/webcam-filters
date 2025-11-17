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
from PIL import Image, ImageTk


# Configuration constants - FFT Filter
DEFAULT_FFT_RADIUS = 30  # Default radius for FFT low-frequency reject circle
DEFAULT_FFT_SMOOTHNESS = 0  # Default smoothness (0 = hard circle, 100 = very smooth transition)
DEFAULT_SHOW_FFT = False  # Default: don't show FFT visualization
DEFAULT_GAIN = 1  # Default gain (0.2 to 5, 1=no change)
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

    def draw_rgb_channels(self, frame, red_params, green_params, blue_params):
        """Apply FFT-based high-pass filter to individual RGB channels

        Args:
            frame: Input BGR frame
            red_params: dict with 'enable', 'radius', 'smoothness'
            green_params: dict with 'enable', 'radius', 'smoothness'
            blue_params: dict with 'enable', 'radius', 'smoothness'
        """
        # Split into BGR channels
        b, g, r = cv2.split(frame)

        # Process each channel - params come in as red, green, blue but we process in BGR order
        # Red channel (index 2 in BGR) - uses red_params
        if red_params['enable']:
            r_filtered = self._apply_fft_to_channel(r, red_params['radius'], red_params['smoothness'])
        else:
            r_filtered = np.zeros_like(r)

        # Green channel (index 1 in BGR) - uses green_params
        if green_params['enable']:
            g_filtered = self._apply_fft_to_channel(g, green_params['radius'], green_params['smoothness'])
        else:
            g_filtered = np.zeros_like(g)

        # Blue channel (index 0 in BGR) - uses blue_params
        if blue_params['enable']:
            b_filtered = self._apply_fft_to_channel(b, blue_params['radius'], blue_params['smoothness'])
        else:
            b_filtered = np.zeros_like(b)

        # Build channels in BGR order for cv2.merge
        channels_processed = [b_filtered, g_filtered, r_filtered]

        # If showing FFT visualization, show all three channels' masks
        if self.show_fft:
            return self._visualize_rgb_fft_masks(frame, red_params, green_params, blue_params)

        # Merge channels back
        result = cv2.merge(channels_processed)
        return result

    def _apply_fft_to_channel(self, channel, radius, smoothness, normalize=True):
        """Apply FFT filter to a single channel

        Args:
            channel: Input channel
            radius: High-pass filter radius (0 = no filtering, but still goes through FFT/IFFT)
            smoothness: Transition smoothness
            normalize: If True, normalize output to 0-255. If False, clamp to 0-255.
        """
        # Compute FFT
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get dimensions
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # Create mask
        center_y, center_x = crow, ccol
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        mask = self._create_mask(distance, radius, smoothness, rows, cols)

        # Apply mask
        fshift = dft_shift * mask

        # Inverse FFT with DFT_SCALE flag to properly normalize the output
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        if normalize:
            # Normalize to 0-255 (stretches min-max range)
            high_pass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Just clamp to 0-255 without stretching
            high_pass = np.clip(img_back, 0, 255).astype(np.uint8)

        return high_pass

    def _create_mask(self, distance, radius, smoothness, rows, cols):
        """Create high-pass mask with given parameters

        Special case: radius=0 means no filtering, pass all frequencies through
        """
        # Special case: radius=0 means no filtering
        if radius == 0:
            # Pass all frequencies (all ones mask)
            return np.ones((rows, cols, 2), np.float32)

        if smoothness == 0:
            # Hard circle mask
            mask = np.ones((rows, cols, 2), np.float32)
            mask_area = distance <= radius
            mask[mask_area] = 0
        else:
            # Smooth transition using sigmoid, but clamped to preserve hard cutoff inside radius
            sigma = smoothness / 10.0
            mask = np.ones((rows, cols, 2), np.float32)
            transition = 1.0 / (1.0 + np.exp(-(distance - radius) / sigma))
            transition = np.clip(transition, 0, 1)
            transition[distance < radius] = 0
            mask[:, :, 0] = transition
            mask[:, :, 1] = transition
        return mask

    def _visualize_grayscale_fft(self, gray_image, radius_list):
        """Visualize FFT of a grayscale image with dashed circles for each radius

        Args:
            gray_image: Grayscale image to compute FFT from
            radius_list: List of radius values to draw as dashed circles

        Returns:
            BGR visualization image
        """
        # Compute FFT
        dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

        # Normalize to 0-255
        normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        vis_gray = normalized.astype(np.uint8)

        # Convert to BGR for drawing colored circles
        vis_bgr = cv2.cvtColor(vis_gray, cv2.COLOR_GRAY2BGR)

        # Draw dashed circles for each unique radius
        rows, cols = gray_image.shape
        center = (cols // 2, rows // 2)

        for radius in sorted(radius_list):
            if radius > 0:
                # Draw dashed circle
                num_dashes = 60
                for i in range(num_dashes):
                    angle1 = (i / num_dashes) * 2 * np.pi
                    angle2 = ((i + 0.5) / num_dashes) * 2 * np.pi
                    x1 = int(center[0] + radius * np.cos(angle1))
                    y1 = int(center[1] + radius * np.sin(angle1))
                    x2 = int(center[0] + radius * np.cos(angle2))
                    y2 = int(center[1] + radius * np.sin(angle2))
                    cv2.line(vis_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow dashed circle

        return vis_bgr

    def _visualize_rgb_fft_masks(self, frame, red_params, green_params, blue_params):
        """Visualize the FFT masks for all RGB channels with colored circles

        Args:
            frame: Input BGR frame (we'll compute FFT from grayscale like composite mode)
            red_params: dict with 'enable', 'radius', 'smoothness'
            green_params: dict with 'enable', 'radius', 'smoothness'
            blue_params: dict with 'enable', 'radius', 'smoothness'
        """
        # Convert to grayscale to compute a single FFT (like composite mode)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute FFT from grayscale
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

        # Normalize to 0-255 (matching grayscale mode behavior)
        magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Count enabled channels to normalize brightness
        num_enabled = sum([blue_params['enable'], green_params['enable'], red_params['enable']])

        # If no channels enabled, just show black
        if num_enabled == 0:
            fft_display = np.zeros((rows, cols, 3), dtype=np.uint8)
        else:
            # Divide brightness by number of enabled channels to prevent overbrightening
            # When all 3 enabled, each gets 1/3 brightness so sum = same as single channel
            magnitude_scaled = (magnitude_normalized.astype(np.float32) / num_enabled).astype(np.uint8)

            # Create color FFT display by colorizing based on enabled channels
            # Start with black
            fft_channels = [np.zeros((rows, cols), dtype=np.uint8),
                           np.zeros((rows, cols), dtype=np.uint8),
                           np.zeros((rows, cols), dtype=np.uint8)]

            # Add scaled grayscale FFT to enabled channels (BGR order)
            if blue_params['enable']:
                fft_channels[0] = magnitude_scaled  # Blue channel
            if green_params['enable']:
                fft_channels[1] = magnitude_scaled  # Green channel
            if red_params['enable']:
                fft_channels[2] = magnitude_scaled  # Red channel

            # Merge the three channels to create color visualization
            fft_display = cv2.merge(fft_channels)  # BGR order

        # Draw colored circles for each enabled channel (BGR order)
        params_list = [(blue_params, (255, 0, 0)),    # Blue in BGR
                       (green_params, (0, 255, 0)),    # Green in BGR
                       (red_params, (0, 0, 255))]      # Red in BGR

        for params, color_bgr in params_list:
            if params['enable']:
                radius = params['radius']
                smoothness = params['smoothness']

                # Calculate opacity based on smoothness
                circle_opacity = 0.5 * (1.0 - smoothness / 100.0)

                # Draw filled circle with variable opacity
                overlay = fft_display.copy()
                cv2.circle(overlay, (ccol, crow), int(radius), color_bgr, -1)
                fft_display = cv2.addWeighted(fft_display, 1.0, overlay, circle_opacity, 0)

                # Draw black dotted line at the radius to show cutoff even when overlapping
                # Draw dotted circle by drawing short arc segments
                num_segments = 60
                for i in range(num_segments):
                    if i % 2 == 0:  # Draw every other segment for dotted effect
                        angle1 = (i / num_segments) * 360
                        angle2 = ((i + 1) / num_segments) * 360
                        cv2.ellipse(fft_display, (ccol, crow), (int(radius), int(radius)),
                                   0, angle1, angle2, (0, 0, 0), 2)

        return fft_display

    def draw_grayscale_bitplanes(self, frame, bitplane_params):
        """Apply FFT-based high-pass filter to individual grayscale bit planes

        Args:
            frame: Input BGR frame
            bitplane_params: List of 8 dicts, each with 'enable', 'radius', 'smoothness'
                            Index 0 = bit 0 (LSB), Index 7 = bit 7 (MSB)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # If showing FFT visualization, determine which image to use for FFT
        if self.show_fft:
            # Count enabled bit planes
            num_enabled = sum(1 for params in bitplane_params if params['enable'])

            if num_enabled == 0:
                # No bits enabled - show black image with no circles
                rows, cols = gray.shape
                return np.zeros((rows, cols, 3), dtype=np.uint8)
            elif num_enabled == 1:
                # Single bit: show FFT of 0-255 scaled bit plane
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        bit_plane = ((gray >> bit) & 1) * 255
                        source_image = bit_plane.astype(np.uint8)
                        break
            else:
                # Multiple bits: show FFT of the masked grayscale image
                bit_mask = 0
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        bit_mask |= (1 << bit)
                source_image = gray & bit_mask

            # Collect all unique radius values from enabled bit planes
            enabled_params = [bitplane_params[i] for i in range(8) if bitplane_params[i]['enable']]
            radius_list = list(set(params['radius'] for params in enabled_params))

            # Visualize FFT of the source image with all radius circles
            return self._visualize_grayscale_fft(source_image, radius_list)

        # Decompose into bit planes
        bit_planes = []
        for bit in range(8):
            # Extract this bit plane (create binary image where this bit is set)
            bit_plane = ((gray >> bit) & 1) * 255
            bit_planes.append(bit_plane.astype(np.uint8))

        # Process each bit plane
        # Note: bitplane_params[i] corresponds to UI row i, where:
        # i=0 is MSB (bit 7), i=7 is LSB (bit 0)
        # So we need to reverse the mapping: bit N uses params[7-N]
        filtered_bit_planes = []
        for bit in range(8):
            ui_index = 7 - bit  # Reverse mapping: bit 7 (MSB) uses params[0], bit 0 (LSB) uses params[7]
            params = bitplane_params[ui_index]

            if params['enable']:
                # Apply FFT filter to this bit plane
                filtered = self._apply_fft_to_channel(bit_planes[bit],
                                                      params['radius'],
                                                      params['smoothness'])
                filtered_bit_planes.append(filtered)
            else:
                # If disabled, keep the original bit plane (no filtering)
                filtered_bit_planes.append(bit_planes[bit])

        # Reconstruct grayscale image from filtered bit planes
        # Count how many bit planes are enabled
        num_enabled = sum(1 for params in bitplane_params if params['enable'])

        if num_enabled == 0:
            # No bit planes enabled - return all black
            return np.zeros_like(gray, dtype=np.uint8)

        elif num_enabled == 1:
            # Single bit enabled: return the filtered bit plane scaled to 0-255
            for ui_index in range(8):
                if bitplane_params[ui_index]['enable']:
                    bit = 7 - ui_index
                    binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                    return binary_plane * 255

        else:
            # Multiple bits enabled: reconstruct grayscale with only those bits, then filter
            # Create bit mask
            bit_mask = 0
            for ui_index in range(8):
                if bitplane_params[ui_index]['enable']:
                    bit = 7 - ui_index
                    bit_mask |= (1 << bit)

            # Apply mask to keep only selected bits
            masked_gray = gray & bit_mask

            # Check if all enabled bits have same radius/smoothness
            enabled_params = [bitplane_params[i] for i in range(8) if bitplane_params[i]['enable']]
            same_params = all(p['radius'] == enabled_params[0]['radius'] and
                            p['smoothness'] == enabled_params[0]['smoothness']
                            for p in enabled_params)

            if same_params:
                # Same filter params - apply FFT to the masked grayscale image
                # Use normalize=False to preserve the gray levels (don't stretch to 0-255)
                filtered = self._apply_fft_to_channel(masked_gray,
                                                      enabled_params[0]['radius'],
                                                      enabled_params[0]['smoothness'],
                                                      normalize=False)

                return filtered
            else:
                # Different params - process each bit separately (old behavior)
                reconstructed = np.zeros_like(gray, dtype=np.uint8)
                for bit in range(8):
                    ui_index = 7 - bit
                    if bitplane_params[ui_index]['enable']:
                        binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                        reconstructed += (binary_plane << bit)
                return reconstructed

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
            # Compute magnitude spectrum for visualization (use ORIGINAL dft_shift to show full spectrum)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

            # Normalize to 0-255
            magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Convert to 3-channel for drawing circle
            fft_display = cv2.cvtColor(magnitude_normalized, cv2.COLOR_GRAY2BGR)

            # Calculate opacity based on smoothness
            # smoothness 0 = 50% opacity, smoothness 100 = 0% opacity (25% at smoothness 50)
            circle_opacity = 0.5 * (1.0 - self.smoothness / 100.0)

            # Draw red circle with variable opacity showing the reject region
            overlay = fft_display.copy()
            cv2.circle(overlay, (ccol, crow), int(self.radius), (0, 0, 255), -1)
            fft_display = cv2.addWeighted(fft_display, 1.0 - circle_opacity, overlay, circle_opacity, 0)

            return fft_display

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        high_pass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return high_pass


class VideoWindow:
    """Tkinter window for displaying video using PIL/ImageTk"""

    def __init__(self, root, title="Video", width=640, height=480):
        """Create a video display window

        Args:
            root: Parent Tkinter root window
            title: Window title
            width: Initial window width
            height: Initial window height
        """
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title(title)

        # Try using Canvas instead of Label for more direct pixel control
        self.canvas = tk.Canvas(self.window, width=width, height=height, highlightthickness=0)
        self.canvas.pack()

        # Store current photo reference to prevent garbage collection
        self.current_photo = None
        self.canvas_image_id = None

        # Position window to the right of the UI window (which is at x=0, width=650)
        self.window.geometry(f"{width}x{height}+670+0")

        # Keyboard handler callback
        self.on_key_callback = None

        # Bind keyboard events
        self.window.bind('<space>', lambda e: self._handle_key(' '))
        self.window.bind('<q>', lambda e: self._handle_key('q'))
        self.window.bind('<Escape>', lambda e: self._handle_key('esc'))

        # Flag to check if window is open
        self.is_open = True
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _handle_key(self, key):
        """Handle keyboard input"""
        if self.on_key_callback:
            self.on_key_callback(key)

    def _on_close(self):
        """Handle window close"""
        self.is_open = False
        self.window.destroy()

    def update_frame(self, frame_bgr):
        """Update the displayed frame

        Args:
            frame_bgr: OpenCV BGR frame (numpy array)
        """
        if not self.is_open:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)

        # Convert to ImageTk
        photo = ImageTk.PhotoImage(image=pil_image)

        # Update canvas
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=photo)

        self.current_photo = photo  # Keep reference to prevent garbage collection

    def set_key_callback(self, callback):
        """Set callback for keyboard events

        Args:
            callback: Function to call with key name ('q', 'esc', ' ')
        """
        self.on_key_callback = callback


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

        # Grayscale bit plane controls (8 bit planes: 7 MSB down to 0 LSB)
        self.bitplane_enable = []
        self.bitplane_radius = []  # Stores the actual radius value (exponential scale)
        self.bitplane_radius_slider = []  # Stores the slider position (linear scale)
        self.bitplane_smoothness = []
        for i in range(8):
            self.bitplane_enable.append(tk.BooleanVar(value=True))
            self.bitplane_radius.append(tk.IntVar(value=DEFAULT_FFT_RADIUS))
            self.bitplane_radius_slider.append(tk.DoubleVar(value=self._radius_to_slider(DEFAULT_FFT_RADIUS)))
            self.bitplane_smoothness.append(tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS))

        self._build_ui()

        # Load saved settings BEFORE adding traces to prevent auto-switching modes
        self._load_settings()

        # Auto-expand bit plane table if grayscale_bitplanes mode is selected
        if self.output_mode.get() == "grayscale_bitplanes":
            self._toggle_bitplane_table()

        # Add traces to auto-select Grayscale Composite when grayscale controls change
        self.fft_radius.trace_add("write", self._on_grayscale_control_change)
        self.fft_smoothness.trace_add("write", self._on_grayscale_control_change)

        # Add traces to auto-select Individual Color Channels when any RGB control changes
        self.red_enable.trace_add("write", self._on_rgb_control_change)
        self.red_radius.trace_add("write", self._on_rgb_control_change)
        self.red_smoothness.trace_add("write", self._on_rgb_control_change)
        self.green_enable.trace_add("write", self._on_rgb_control_change)
        self.green_radius.trace_add("write", self._on_rgb_control_change)
        self.green_smoothness.trace_add("write", self._on_rgb_control_change)
        self.blue_enable.trace_add("write", self._on_rgb_control_change)
        self.blue_radius.trace_add("write", self._on_rgb_control_change)
        self.blue_smoothness.trace_add("write", self._on_rgb_control_change)

        # Add traces to auto-select Grayscale Bit Planes when any bit plane control changes
        for i in range(8):
            self.bitplane_enable[i].trace_add("write", self._on_bitplane_control_change)
            self.bitplane_radius[i].trace_add("write", self._on_bitplane_control_change)
            self.bitplane_smoothness[i].trace_add("write", self._on_bitplane_control_change)

        # Auto-size window to fit all content
        self.root.update_idletasks()  # Ensure all widgets are laid out
        width = 650  # Fixed width (wider for table layout)
        # Get the required height from the container
        height = self.root.winfo_reqheight()
        # Add a small buffer
        height = min(height + 20, 1000)  # Cap at 1000px
        # Position UI window at top-left, video window will be to the right at x=670
        self.root.geometry(f"{width}x{height}+0+0")

        # Flag to check if window is still open
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Make sure window is visible and brought to front
        self.root.deiconify()
        self.root.lift()

    def _slider_to_radius(self, slider_value):
        """Convert linear slider value (0-100) to exponential radius (0-200+)

        Uses formula: radius = floor(e^(slider/20) - 1)
        This gives fine control at low values and larger steps at high values
        """
        import math
        return int(math.exp(slider_value / 20.0) - 1)

    def _radius_to_slider(self, radius):
        """Convert exponential radius value to linear slider position

        Inverse of _slider_to_radius: slider = 20 * ln(radius + 1)
        """
        import math
        if radius <= 0:
            return 0
        return 20.0 * math.log(radius + 1)

    def _build_ui(self):
        """Build the tkinter UI"""
        padding = {'padx': 10, 'pady': 3}

        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(self.root, height=700)  # Set reasonable initial height
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
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

        # Use scrollable_frame as container instead of root
        container = scrollable_frame

        # Bind mousewheel scrolling - store canvas reference for the callback
        self._scrollable_canvas = canvas

        def _on_mousewheel(event):
            # Platform-specific scrolling
            if event.num == 5 or event.delta < 0:
                self._scrollable_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                self._scrollable_canvas.yview_scroll(-1, "units")
            return "break"

        # Bind for macOS/Windows
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        # Bind for Linux
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

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

        # Show FFT checkbox - placed before the output mode sections
        ttk.Checkbutton(fft_frame, text="Show FFT (instead of image)", variable=self.show_fft).pack(anchor='w', pady=(5, 10))

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

        # Row 2: Individual Color Channels - grouped section with table layout
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

        # Row 3: Grayscale Bit Planes - grouped section with table layout
        gs_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        gs_bitplanes_group.pack(fill='x', pady=(0, 10))

        # Header with radio button and expand/collapse button
        header_frame = ttk.Frame(gs_bitplanes_group)
        header_frame.pack(fill='x', pady=(0, 5))

        # Radio button on left
        bitplane_radio = ttk.Radiobutton(header_frame, text="Grayscale Bit Planes",
                                        value="grayscale_bitplanes",
                                        variable=self.output_mode,
                                        command=self._on_bitplane_radio_select)
        bitplane_radio.pack(side='left', anchor='w')

        # Expand/collapse on right
        self.bitplane_expanded = tk.BooleanVar(value=False)
        self.bitplane_toggle_btn = ttk.Button(header_frame, text="▶", width=1,
                                              command=self._toggle_bitplane_table)
        self.bitplane_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create table for bit planes (initially hidden)
        bitplane_table_frame = ttk.Frame(gs_bitplanes_group)
        self.bitplane_table_frame = bitplane_table_frame  # Store reference for show/hide

        # Header row (row 0)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        ttk.Label(bitplane_table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(bitplane_table_frame, text="Smoothness (Sigmoid/10)").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=5, padx=2, pady=2)

        # Create 8 rows for bit planes (7 MSB down to 0 LSB)
        bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]
        for i, label in enumerate(bit_labels):
            row = i + 1  # +1 for header row

            # Bit plane label (right-justified)
            ttk.Label(bitplane_table_frame, text=label).grid(row=row, column=0, padx=5, pady=5, sticky='e')

            # Enabled checkbox
            ttk.Checkbutton(bitplane_table_frame, variable=self.bitplane_enable[i]).grid(row=row, column=1, padx=5, pady=5)

            # Radius slider (exponential scale: 0-100 slider -> 0-200+ radius)
            def update_radius(slider_val, idx=i):
                radius = self._slider_to_radius(float(slider_val))
                self.bitplane_radius[idx].set(radius)

            radius_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_radius_slider[i], orient='horizontal',
                                     command=lambda v, idx=i: update_radius(v, idx))
            radius_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')

            # Radius value label
            tk.Label(bitplane_table_frame, textvariable=self.bitplane_radius[i], width=4).grid(row=row, column=3, padx=(2, 10), pady=5)

            # Smoothness slider
            smooth_slider = ttk.Scale(bitplane_table_frame, from_=0, to=100, variable=self.bitplane_smoothness[i], orient='horizontal',
                                      command=lambda v, idx=i: self.bitplane_smoothness[idx].set(int(float(v))))
            smooth_slider.grid(row=row, column=4, padx=5, pady=5, sticky='ew')

            # Smoothness value label
            tk.Label(bitplane_table_frame, textvariable=self.bitplane_smoothness[i], width=4).grid(row=row, column=5, padx=2, pady=5)

        # Configure column weights for proper expansion
        bitplane_table_frame.columnconfigure(2, weight=1)  # Filter Radius column expands
        bitplane_table_frame.columnconfigure(4, weight=1)  # Smoothness column expands

        # Row 4: Color Bitplanes - grouped section
        color_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        color_bitplanes_group.pack(fill='x', pady=(0, 10))

        ttk.Radiobutton(color_bitplanes_group, text="Color Bitplanes (Not Implemented)", value="color_bitplanes",
                       variable=self.output_mode, state='disabled').pack(anchor='w')

        # Common controls at bottom
        common_frame = ttk.Frame(fft_frame)
        common_frame.pack(fill='x', pady=(10, 0))

        # Gain slider with value on right (log scale: 0.2 to 5, center=1)
        ttk.Label(common_frame, text="Gain (1/5x to 1x [no change] to 5x)").pack(anchor='w', pady=(5, 0))
        gain_row = ttk.Frame(common_frame)
        gain_row.pack(fill='x')
        gain_slider = ttk.Scale(gain_row, from_=0.2, to=5,
                               variable=self.gain, orient='horizontal',
                               command=lambda v: self.gain_display.set(f"{float(v):.2f}"))
        gain_slider.pack(side='left', fill='x', expand=True)
        ttk.Label(gain_row, textvariable=self.gain_display, width=6).pack(side='left', padx=(5, 0))

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
            'bitplane_smoothness': [bp.get() for bp in self.bitplane_smoothness]
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
            gain_value = settings.get('gain', DEFAULT_GAIN)
            self.gain.set(gain_value)
            self.gain_display.set(f"{gain_value:.1f}")  # Update display to match loaded value
            self.invert.set(settings.get('invert', DEFAULT_INVERT))
            self.output_mode.set(settings.get('output_mode', DEFAULT_OUTPUT_MODE))

            # Load RGB channel settings
            self.red_enable.set(settings.get('red_enable', True))
            self.red_radius.set(settings.get('red_radius', DEFAULT_FFT_RADIUS))
            self.red_smoothness.set(settings.get('red_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.green_enable.set(settings.get('green_enable', True))
            self.green_radius.set(settings.get('green_radius', DEFAULT_FFT_RADIUS))
            self.green_smoothness.set(settings.get('green_smoothness', DEFAULT_FFT_SMOOTHNESS))
            self.blue_enable.set(settings.get('blue_enable', True))
            self.blue_radius.set(settings.get('blue_radius', DEFAULT_FFT_RADIUS))
            self.blue_smoothness.set(settings.get('blue_smoothness', DEFAULT_FFT_SMOOTHNESS))

            # Load grayscale bit plane settings
            bitplane_enable = settings.get('bitplane_enable', [True] * 8)
            bitplane_radius = settings.get('bitplane_radius', [DEFAULT_FFT_RADIUS] * 8)
            bitplane_smoothness = settings.get('bitplane_smoothness', [DEFAULT_FFT_SMOOTHNESS] * 8)
            for i in range(8):
                self.bitplane_enable[i].set(bitplane_enable[i] if i < len(bitplane_enable) else True)

                # Set radius and update slider position to match
                radius_val = bitplane_radius[i] if i < len(bitplane_radius) else DEFAULT_FFT_RADIUS
                self.bitplane_radius[i].set(radius_val)
                self.bitplane_radius_slider[i].set(self._radius_to_slider(radius_val))

                self.bitplane_smoothness[i].set(bitplane_smoothness[i] if i < len(bitplane_smoothness) else DEFAULT_FFT_SMOOTHNESS)

            print(f"Settings loaded from {settings_file}")
        except Exception as e:
            print(f"Error loading settings: {e}")

    def _on_rgb_control_change(self, *args):
        """Auto-select Individual Color Channels radio button when RGB controls are changed"""
        self.output_mode.set("color_channels")

    def _on_grayscale_control_change(self, *args):
        """Auto-select Grayscale Composite radio button when grayscale controls are changed"""
        self.output_mode.set("grayscale_composite")

    def _on_bitplane_control_change(self, *args):
        """Auto-select Grayscale Bit Planes radio button when bit plane controls are changed"""
        self.output_mode.set("grayscale_bitplanes")
        # Also expand the table when a control is changed
        if not self.bitplane_expanded.get():
            self._toggle_bitplane_table()

    def _toggle_bitplane_table(self):
        """Toggle the visibility of the bit plane table"""
        if self.bitplane_expanded.get():
            # Collapse
            self.bitplane_table_frame.pack_forget()
            self.bitplane_toggle_btn.config(text="▶")
            self.bitplane_expanded.set(False)
        else:
            # Expand
            self.bitplane_table_frame.pack(fill='x', padx=10)
            self.bitplane_toggle_btn.config(text="▼")
            self.bitplane_expanded.set(True)
            # Also select the radio button when expanding
            self.output_mode.set("grayscale_bitplanes")

    def _on_bitplane_radio_select(self):
        """Expand the bit plane table when radio button is selected"""
        if not self.bitplane_expanded.get():
            self._toggle_bitplane_table()

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
        # Apply FFT filter
        fft_result = self.fft.draw(frame)

        # If showing FFT, it's already BGR, otherwise convert from grayscale
        if self.fft.show_fft:
            return fft_result
        else:
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

    # Initialize combined sketch filter
    sketch = CombinedSketchFilter(width, height)

    # Create tkinter control panel
    controls = ControlPanel(width, height, available_cameras, selected_camera['id'])

    # Create Tkinter video window
    video_window = VideoWindow(controls.root, title='Sketch Filter', width=width, height=height)

    # Set up keyboard handler for video window
    def handle_video_key(key):
        if key in ['q', 'esc']:
            controls.running = False
        elif key == ' ':
            # Toggle effect
            controls.show_original.set(not controls.show_original.get())

    video_window.set_key_callback(handle_video_key)

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

            # Get output mode
            output_mode = controls.output_mode.get()

            if output_mode == "color_channels":
                # Individual Color Channels mode - use RGB channel filtering
                red_params = {
                    'enable': controls.red_enable.get(),
                    'radius': controls.red_radius.get(),
                    'smoothness': controls.red_smoothness.get()
                }
                green_params = {
                    'enable': controls.green_enable.get(),
                    'radius': controls.green_radius.get(),
                    'smoothness': controls.green_smoothness.get()
                }
                blue_params = {
                    'enable': controls.blue_enable.get(),
                    'radius': controls.blue_radius.get(),
                    'smoothness': controls.blue_smoothness.get()
                }
                # Pass params in RGB order (function signature order)
                result = sketch.fft.draw_rgb_channels(frame, red_params, green_params, blue_params)
            elif output_mode == "grayscale_bitplanes":
                # Grayscale Bit Planes mode - filter each bit plane independently
                bitplane_params = []
                for i in range(8):
                    bitplane_params.append({
                        'enable': controls.bitplane_enable[i].get(),
                        'radius': controls.bitplane_radius[i].get(),
                        'smoothness': controls.bitplane_smoothness[i].get()
                    })
                gray_result = sketch.fft.draw_grayscale_bitplanes(frame, bitplane_params)
                # Convert grayscale to BGR for display (unless already BGR from FFT visualization)
                if len(gray_result.shape) == 2:
                    result = cv2.cvtColor(gray_result, cv2.COLOR_GRAY2BGR)
                else:
                    result = gray_result
            else:
                # Grayscale composite mode (and other modes to be implemented)
                result = sketch.draw(frame)

            # Apply gain and invert only if NOT showing FFT
            if not controls.show_fft.get():
                # Apply gain (multiplicative, centered at 1)
                # gain = 1: no change
                # gain = 5: 5x brighter
                # gain = 0.2: 1/5 as bright (darker)
                gain = controls.gain.get()
                if gain != 1:
                    result = np.clip(result * gain, 0, 255).astype(np.uint8)

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

        # Debug: Save BGR result before display (once)
        if output_mode == "grayscale_bitplanes" and not hasattr(sketch, '_bgr_saved'):
            sketch._bgr_saved = True
            cv2.imwrite('/tmp/bgr_before_display.png', result)
            print(f"BGR result saved to: /tmp/bgr_before_display.png")
            print(f"BGR result shape: {result.shape}, dtype: {result.dtype}")
            # Sample the box regions
            h, w = result.shape[:2]
            box_size = 80
            x_start = w - 4 * box_size - 40
            y_start = 20
            for i, expected_val in enumerate([0, 64, 128, 192]):
                x = x_start + i * box_size
                box_region = result[y_start:y_start+box_size, x:x+box_size, 0]  # Sample B channel
                unique_vals = np.unique(box_region)
                print(f"BGR Box {i}: Expected={expected_val}, Unique values={unique_vals}")

        # Show the result using Tkinter video window
        video_window.update_frame(result)

        # Check if video window was closed
        if not video_window.is_open:
            break

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

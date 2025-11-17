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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend first
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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

        # Matplotlib visualization window
        self.viz_window = None
        self.viz_fig = None
        self.viz_ax = None
        self.viz_canvas = None

    def update(self):
        """Update - not needed for static effect"""
        pass

    def create_visualization_window(self, control_panel_y=0, control_panel_height=800):
        """Create a matplotlib window to visualize the filter curve

        Args:
            control_panel_y: Y position of control panel window
            control_panel_height: Height of control panel window
        """
        if self.viz_window is not None:
            return  # Already created

        # Create Tkinter window for matplotlib
        self.viz_window = tk.Toplevel()
        self.viz_window.title("Filter Curve Visualization")
        # Position will be set by the caller after all windows are created
        self.viz_window.geometry("600x400+0+0")  # Temporary position

        # Create matplotlib figure using Figure directly (not plt.subplots)
        self.viz_fig = Figure(figsize=(6, 4), dpi=100)
        self.viz_ax = self.viz_fig.add_subplot(111)
        self.viz_ax.set_xlabel('Distance from Center (pixels)')
        self.viz_ax.set_ylabel('Mask Value (0=blocked, 1=passed)')
        self.viz_ax.set_title('FFT Filter Transition Curve')
        self.viz_ax.grid(True, alpha=0.3)
        self.viz_ax.set_ylim(-0.1, 1.1)

        # Embed matplotlib in Tkinter
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.viz_window)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Handle window close
        self.viz_window.protocol("WM_DELETE_WINDOW", self._on_viz_close)

    def _on_viz_close(self):
        """Handle visualization window close"""
        if self.viz_window:
            self.viz_window.destroy()
            self.viz_window = None
            self.viz_fig = None
            self.viz_ax = None
            self.viz_canvas = None

    def update_visualization(self, mask, distance, radius, smoothness, rgb_params=None, bitplane_params=None, color_bitplane_params=None):
        """Update the filter curve visualization

        Args:
            mask: The frequency domain mask (rows, cols, 2)
            distance: Distance array from center
            radius: Current filter radius
            smoothness: Current smoothness value
            rgb_params: Optional dict with 'red', 'green', 'blue' params for RGB mode
            bitplane_params: Optional list of 8 dicts with 'enable', 'radius', 'smoothness' for bit plane mode
            color_bitplane_params: Optional dict with keys 'red', 'green', 'blue',
                                  each containing list of 8 dicts with 'enable', 'radius', 'smoothness'
        """
        if self.viz_window is None:
            return

        try:
            # Get horizontal cross-section through center
            rows, cols = distance.shape
            center_y = rows // 2

            # Get distances along horizontal center line
            distances = distance[center_y, :]

            # Clear and redraw
            self.viz_ax.clear()
            self.viz_ax.set_xlabel('Distance from FFT Center (pixels)', fontsize=10)
            self.viz_ax.set_ylabel('Mask Value (0=blocked, 1=passed)', fontsize=10)
            self.viz_ax.grid(True, alpha=0.3)
            self.viz_ax.set_ylim(-0.1, 1.1)
            self.viz_ax.set_xlim(0, 400)

            # Draw horizontal lines at 0, 0.03, 0.5, and 1
            self.viz_ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            self.viz_ax.axhline(y=0.03, color='gray', linestyle=':', alpha=0.3)
            self.viz_ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
            self.viz_ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

            if color_bitplane_params is not None:
                # Color Bit Planes mode - plot bit plane curves for each color channel
                self.viz_ax.set_title('Color Bit Plane Filter Responses', fontsize=12, fontweight='bold')

                # Use varying linewidths and line styles to distinguish bit planes
                # Bits 7-4: Solid lines with decreasing width
                # Bits 3-0: Dotted lines with same widths as 7-4
                linewidths = [
                    2.0,  # Bit 7 (MSB): thickest solid
                    1.7,  # Bit 6: solid
                    1.5,  # Bit 5: solid
                    1.3,  # Bit 4: solid
                    2.0,  # Bit 3: thickest dotted (same as bit 7)
                    1.7,  # Bit 2: dotted (same as bit 6)
                    1.5,  # Bit 1: dotted (same as bit 5)
                    1.3,  # Bit 0 (LSB): dotted (same as bit 4)
                ]

                # Line styles: solid for bits 7-4, dotted for bits 3-0
                linestyles = ['-', '-', '-', '-', ':', ':', ':', ':']

                # Use slightly varying alpha values to help distinguish overlapping lines
                alphas = [0.95, 0.90, 0.85, 0.80, 0.95, 0.90, 0.85, 0.80]

                # Colors for each channel: red, green, blue
                colors = {'red': 'r', 'green': 'g', 'blue': 'b'}

                # Plot each color's bit planes
                for color_name, color_code in colors.items():
                    bitplane_params = color_bitplane_params[color_name]

                    # Plot each enabled bit plane for this color
                    for i in range(8):
                        if bitplane_params[i]['enable']:
                            bit_mask = self._compute_filter_curve(distances, bitplane_params[i]['radius'], bitplane_params[i]['smoothness'])
                            bit_label = f"{color_name.capitalize()} Bit {7-i}" if i == 0 else f"{color_name[0].upper()}{7-i}"
                            if i == 0:
                                bit_label = f"{color_name.capitalize()} Bit 7 (MSB)"
                            elif i == 7:
                                bit_label = f"{color_name.capitalize()} Bit 0 (LSB)"
                            else:
                                bit_label = f"{color_name.capitalize()} Bit {7-i}"
                            # Use varying linewidths and solid vs dotted for distinction, colored by channel
                            self.viz_ax.plot(distances, bit_mask, color=color_code,
                                           linewidth=linewidths[i], linestyle=linestyles[i],
                                           label=bit_label, alpha=alphas[i], antialiased=True)

                            # Draw vertical line at this bit plane's radius
                            if bitplane_params[i]['radius'] > 0:
                                self.viz_ax.axvline(x=bitplane_params[i]['radius'], color=color_code,
                                                  linestyle=':', linewidth=1.0, alpha=0.4)

                # Add legend with smaller font and multiple columns
                self.viz_ax.legend(loc='lower right', fontsize=7, ncol=3)
            elif rgb_params is not None:
                # RGB mode - plot separate curves for each enabled channel
                self.viz_ax.set_title('RGB Channel Filter Responses', fontsize=12, fontweight='bold')

                # Use line widths from bit plane pattern (top 3: 2.0, 1.7, 1.5)
                # All solid lines, but with R, G, B colors
                # Red channel - thickest (2.0)
                if rgb_params['red']['enable']:
                    red_mask = self._compute_filter_curve(distances, rgb_params['red']['radius'], rgb_params['red']['smoothness'])
                    self.viz_ax.plot(distances, red_mask, 'r-', linewidth=2.0, label='Red', alpha=0.95)

                # Green channel - medium (1.7)
                if rgb_params['green']['enable']:
                    green_mask = self._compute_filter_curve(distances, rgb_params['green']['radius'], rgb_params['green']['smoothness'])
                    self.viz_ax.plot(distances, green_mask, 'g-', linewidth=1.7, label='Green', alpha=0.90)

                # Blue channel - thinnest (1.5)
                if rgb_params['blue']['enable']:
                    blue_mask = self._compute_filter_curve(distances, rgb_params['blue']['radius'], rgb_params['blue']['smoothness'])
                    self.viz_ax.plot(distances, blue_mask, 'b-', linewidth=1.5, label='Blue', alpha=0.85)

                # Add legend
                self.viz_ax.legend(loc='lower right', fontsize=9)
            elif bitplane_params is not None:
                # Bit plane mode - plot curves for each enabled bit plane
                self.viz_ax.set_title('Bit Plane Filter Responses', fontsize=12, fontweight='bold')

                # Use varying linewidths and line styles to distinguish bit planes
                # Bits 7-4: Solid lines with decreasing width
                # Bits 3-0: Dotted lines with same widths as 7-4
                linewidths = [
                    2.0,  # Bit 7 (MSB): thickest solid
                    1.7,  # Bit 6: solid
                    1.5,  # Bit 5: solid
                    1.3,  # Bit 4: solid
                    2.0,  # Bit 3: thickest dotted (same as bit 7)
                    1.7,  # Bit 2: dotted (same as bit 6)
                    1.5,  # Bit 1: dotted (same as bit 5)
                    1.3,  # Bit 0 (LSB): dotted (same as bit 4)
                ]

                # Line styles: solid for bits 7-4, dotted for bits 3-0
                linestyles = ['-', '-', '-', '-', ':', ':', ':', ':']

                # Use slightly varying alpha values to help distinguish overlapping lines
                alphas = [0.95, 0.90, 0.85, 0.80, 0.95, 0.90, 0.85, 0.80]

                # Plot each enabled bit plane
                for i in range(8):
                    if bitplane_params[i]['enable']:
                        bit_mask = self._compute_filter_curve(distances, bitplane_params[i]['radius'], bitplane_params[i]['smoothness'])
                        bit_label = f"Bit {7-i}" if i == 0 else f"{7-i}"
                        if i == 0:
                            bit_label = "Bit 7 (MSB)"
                        elif i == 7:
                            bit_label = "Bit 0 (LSB)"
                        else:
                            bit_label = f"Bit {7-i}"
                        # Use varying linewidths and solid vs dotted for distinction
                        self.viz_ax.plot(distances, bit_mask, color='black',
                                       linewidth=linewidths[i], linestyle=linestyles[i],
                                       label=bit_label, alpha=alphas[i], antialiased=True)

                # Add legend
                self.viz_ax.legend(loc='lower right', fontsize=8, ncol=2)
            else:
                # Grayscale mode - single black curve
                self.viz_ax.set_title('Butterworth Highpass Filter Response', fontsize=12, fontweight='bold')

                # Get mask values along horizontal center line
                mask_values = mask[center_y, :, 0]

                # Calculate Butterworth order from smoothness
                # Map smoothness (0-100) to order (0.5-10), inverted
                order = 10.0 - (smoothness / 100.0) * 9.5
                if order < 0.5:
                    order = 0.5

                # Calculate effective cutoff (shifted D0)
                target_attenuation = 0.03
                shift_factor = np.power(1.0/target_attenuation - 1.0, 1.0 / (2.0 * order))
                effective_cutoff = radius * shift_factor

                # Plot the filter curve in black
                self.viz_ax.plot(distances, mask_values, 'k-', linewidth=2)

                # Draw vertical line at user's radius (where filter ≈ 0.03) in red
                self.viz_ax.axvline(x=radius, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='User Radius (H≈0.03)')

                # Draw vertical line at effective D₀ (where filter = 0.5) in blue
                self.viz_ax.axvline(x=effective_cutoff, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='D₀ (H=0.5)')

                # Add legend
                self.viz_ax.legend(loc='lower right', fontsize=8)

                # Add text annotations for parameters
                self.viz_ax.text(0.02, 0.98, f'User Radius: {radius:.1f}',
                               transform=self.viz_ax.transAxes,
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                self.viz_ax.text(0.02, 0.88, f'D₀: {effective_cutoff:.1f}',
                               transform=self.viz_ax.transAxes,
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                self.viz_ax.text(0.02, 0.78, f'Order (n): {order:.2f}',
                               transform=self.viz_ax.transAxes,
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            # Force immediate update
            self.viz_canvas.draw()
        except Exception as e:
            print(f"Error updating visualization: {e}")

    def _compute_filter_curve(self, distances, radius, smoothness):
        """Compute the filter response curve for given parameters

        Args:
            distances: Array of distances from center
            radius: Filter radius
            smoothness: Smoothness parameter (0-100)

        Returns:
            Array of filter response values
        """
        if radius == 0:
            return np.ones_like(distances)

        if smoothness == 0:
            # Hard cutoff
            return (distances > radius).astype(float)

        # Butterworth filter
        order = 10.0 - (smoothness / 100.0) * 9.5
        if order < 0.5:
            order = 0.5

        target_attenuation = 0.03
        shift_factor = np.power(1.0/target_attenuation - 1.0, 1.0 / (2.0 * order))
        effective_cutoff = radius * shift_factor

        with np.errstate(divide='ignore', invalid='ignore'):
            transition = 1.0 / (1.0 + np.power(effective_cutoff / (distances + 1e-10), 2 * order))
            transition = np.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
            transition = np.clip(transition, 0, 1)

        return transition

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

        # Get dimensions for distance calculation
        rows, cols = b.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Update visualization with RGB parameters
        rgb_viz_params = {
            'red': red_params,
            'green': green_params,
            'blue': blue_params
        }
        # Use a dummy mask since we're in RGB mode - the visualization will compute curves
        dummy_mask = np.ones((rows, cols, 2), np.float32)
        self.update_visualization(dummy_mask, distance, 0, 0, rgb_params=rgb_viz_params)

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

        # Store last mask for visualization
        self.last_mask = mask
        self.last_mask_distance = distance
        self.last_mask_radius = radius

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
            # Smooth transition using Butterworth highpass filter
            # Formula: H = 1 / [1 + (D0/D)^(2n)]
            # where D0 = cutoff radius, D = distance, n = order
            # Higher smoothness = lower order = gentler transition
            # Map smoothness (0-100) to order (0.5-10)
            # Lower order = smoother, higher order = sharper
            # Invert so higher smoothness = smoother transition
            order = 10.0 - (smoothness / 100.0) * 9.5
            if order < 0.5:
                order = 0.5

            mask = np.ones((rows, cols, 2), np.float32)

            # Shift D0 to the right so that at distance=radius, H is very small (< 0.03)
            # We want: 0.03 = 1 / [1 + (D0/radius)^(2n)]
            # Solving: (D0/radius)^(2n) = 1/0.03 - 1 = 32.33
            # D0/radius = 32.33^(1/(2n))
            # D0 = radius * 32.33^(1/(2n))
            target_attenuation = 0.03  # Target value at user's radius (within 3% of zero)
            shift_factor = np.power(1.0/target_attenuation - 1.0, 1.0 / (2.0 * order))
            effective_cutoff = radius * shift_factor

            # Butterworth highpass filter formula
            # Avoid division by zero at center
            with np.errstate(divide='ignore', invalid='ignore'):
                # H = 1 / [1 + (D0/D)^(2n)]
                transition = 1.0 / (1.0 + np.power(effective_cutoff / (distance + 1e-10), 2 * order))
                transition = np.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)
                transition = np.clip(transition, 0, 1)

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

        # Get dimensions for distance calculation
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Update visualization with bit plane parameters
        # Use a dummy mask since we're in bitplane mode - the visualization will compute curves
        dummy_mask = np.ones((rows, cols, 2), np.float32)
        self.update_visualization(dummy_mask, distance, 0, 0, bitplane_params=bitplane_params)

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

    def _visualize_color_bitplanes_fft(self, frame, color_bitplane_params):
        """Visualize FFT for color bit planes mode with colored circles for each channel's bit planes

        Args:
            frame: Input BGR frame
            color_bitplane_params: Dict with keys 'red', 'green', 'blue'
                                  Each value is a list of 8 dicts with 'enable', 'radius', 'smoothness'

        Returns:
            BGR visualization image
        """
        # Split into BGR channels
        b, g, r = cv2.split(frame)
        channels = {'blue': b, 'green': g, 'red': r}
        rows, cols = b.shape
        crow, ccol = rows // 2, cols // 2

        # Calculate distance array for visualization
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Update filter curve visualization with color bitplane parameters
        dummy_mask = np.ones((rows, cols, 2), np.float32)
        self.update_visualization(dummy_mask, distance, 0, 0, color_bitplane_params=color_bitplane_params)

        # Process each color channel to determine what to show in FFT
        fft_channels_bgr = []
        for color_name in ['blue', 'green', 'red']:
            channel = channels[color_name]
            bitplane_params = color_bitplane_params[color_name]

            # Count enabled bit planes for this channel
            num_enabled = sum(1 for params in bitplane_params if params['enable'])

            if num_enabled == 0:
                # No bits enabled for this channel - show black
                fft_channels_bgr.append(np.zeros((rows, cols), dtype=np.uint8))
            elif num_enabled == 1:
                # Single bit: show FFT of 0-255 scaled bit plane
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        bit_plane = ((channel >> bit) & 1) * 255
                        source_image = bit_plane.astype(np.uint8)
                        break

                # Compute FFT of this binary bit plane
                dft = cv2.dft(np.float32(source_image), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
                normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                fft_channels_bgr.append(normalized.astype(np.uint8))
            else:
                # Multiple bits: show FFT of the masked channel (analog values)
                bit_mask = 0
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        bit_mask |= (1 << bit)
                source_image = channel & bit_mask

                # Compute FFT of this masked channel
                dft = cv2.dft(np.float32(source_image), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
                normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
                fft_channels_bgr.append(normalized.astype(np.uint8))

        # Merge the three FFT channels
        fft_display = cv2.merge(fft_channels_bgr)

        # Draw circles for each color's bit planes
        # BGR order: Blue, Green, Red
        color_info = [
            ('blue', (255, 128, 0)),    # Bright cyan in BGR for visibility
            ('green', (0, 255, 0)),     # Green in BGR
            ('red', (128, 0, 255))      # Bright magenta in BGR for visibility
        ]

        for color_name, color_bgr in color_info:
            bitplane_params = color_bitplane_params[color_name]

            # Collect all unique radius values from enabled bit planes for this color
            radius_set = set()
            for params in bitplane_params:
                if params['enable'] and params['radius'] > 0:
                    radius_set.add(params['radius'])

            # Draw dotted circles for each unique radius in this color
            for radius in sorted(radius_set):
                # Draw dotted circle using line segments for better visibility
                num_dashes = 80
                for i in range(num_dashes):
                    if i % 2 == 0:  # Draw every other segment for dotted effect
                        angle1 = (i / num_dashes) * 2 * np.pi
                        angle2 = ((i + 1) / num_dashes) * 2 * np.pi
                        x1 = int(ccol + radius * np.cos(angle1))
                        y1 = int(crow + radius * np.sin(angle1))
                        x2 = int(ccol + radius * np.cos(angle2))
                        y2 = int(crow + radius * np.sin(angle2))
                        cv2.line(fft_display, (x1, y1), (x2, y2), color_bgr, 2)

        return fft_display

    def draw_color_bitplanes(self, frame, color_bitplane_params):
        """Apply FFT-based high-pass filter to individual bit planes of each color channel

        Args:
            frame: Input BGR frame
            color_bitplane_params: Dict with keys 'red', 'green', 'blue'
                                  Each value is a list of 8 dicts with 'enable', 'radius', 'smoothness'
                                  Index 0 = bit 0 (LSB), Index 7 = bit 7 (MSB)
        """
        # Update filter curve visualization with color bitplane parameters
        # Calculate distance array for visualization
        rows, cols = frame.shape[:2]
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        dummy_mask = np.ones((rows, cols, 2), np.float32)
        self.update_visualization(dummy_mask, distance, 0, 0, color_bitplane_params=color_bitplane_params)

        # If showing FFT visualization, create and return the visualization
        if self.show_fft:
            return self._visualize_color_bitplanes_fft(frame, color_bitplane_params)

        # Split into BGR channels
        b, g, r = cv2.split(frame)

        # Process each color channel separately
        channels = {'blue': b, 'green': g, 'red': r}
        filtered_channels = {}

        for color_name, channel in channels.items():
            # Get bit plane parameters for this color
            bitplane_params = color_bitplane_params[color_name]

            # Decompose this channel into 8 bit planes
            bit_planes = []
            for bit in range(8):
                # Extract this bit plane (create binary image where this bit is set)
                bit_plane = ((channel >> bit) & 1) * 255
                bit_planes.append(bit_plane.astype(np.uint8))

            # Process each bit plane
            # Note: bitplane_params[i] corresponds to UI row i, where:
            # i=0 is MSB (bit 7), i=7 is LSB (bit 0)
            # So we need to reverse the mapping: bit N uses params[7-N]
            filtered_bit_planes = []
            for bit in range(8):
                ui_index = 7 - bit  # Reverse mapping
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

            # Reconstruct this color channel from filtered bit planes
            # Count how many bit planes are enabled
            num_enabled = sum(1 for params in bitplane_params if params['enable'])

            if num_enabled == 0:
                # No bit planes enabled - return all black
                filtered_channels[color_name] = np.zeros_like(channel, dtype=np.uint8)

            elif num_enabled == 1:
                # Single bit enabled: return the filtered bit plane scaled to 0-255
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                        filtered_channels[color_name] = binary_plane * 255
                        break

            else:
                # Multiple bits enabled: reconstruct channel with only those bits, then filter
                # Create bit mask
                bit_mask = 0
                for ui_index in range(8):
                    if bitplane_params[ui_index]['enable']:
                        bit = 7 - ui_index
                        bit_mask |= (1 << bit)

                # Apply mask to keep only selected bits
                masked_channel = channel & bit_mask

                # Check if all enabled bits have same radius/smoothness
                enabled_params = [bitplane_params[i] for i in range(8) if bitplane_params[i]['enable']]
                same_params = all(p['radius'] == enabled_params[0]['radius'] and
                                p['smoothness'] == enabled_params[0]['smoothness']
                                for p in enabled_params)

                if same_params:
                    # Same filter params - apply FFT to the masked channel
                    # Use normalize=False to preserve the intensity levels
                    filtered = self._apply_fft_to_channel(masked_channel,
                                                          enabled_params[0]['radius'],
                                                          enabled_params[0]['smoothness'],
                                                          normalize=False)
                    filtered_channels[color_name] = filtered
                else:
                    # Different params - process each bit separately
                    reconstructed = np.zeros_like(channel, dtype=np.uint8)
                    for bit in range(8):
                        ui_index = 7 - bit
                        if bitplane_params[ui_index]['enable']:
                            binary_plane = (filtered_bit_planes[bit] > 128).astype(np.uint8)
                            reconstructed += (binary_plane << bit)
                    filtered_channels[color_name] = reconstructed

        # Merge the three filtered color channels back into BGR image
        result = cv2.merge([filtered_channels['blue'],
                           filtered_channels['green'],
                           filtered_channels['red']])
        return result

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

        # Use _create_mask to get the same smooth filtering as other modes
        mask = self._create_mask(distance, self.radius, self.smoothness, rows, cols)

        # Update visualization for grayscale composite mode
        self.update_visualization(mask, distance, self.radius, self.smoothness)

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

        # Position will be set by main() after all windows are created
        self.window.geometry(f"{width}x{height}+0+0")

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

    def resize(self, width, height):
        """Resize the video window

        Args:
            width: New window width
            height: New window height
        """
        if not self.is_open:
            return

        # Resize canvas
        self.canvas.config(width=width, height=height)

        # Update window geometry
        self.window.geometry(f"{width}x{height}")

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
        self.selected_resolution = tk.StringVar(value="")  # Will be set during UI build
        self.resolution_changed = False
        self.new_resolution = None

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

        # Color bit plane controls (3 colors × 8 bit planes each)
        self.color_bitplane_enable = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_radius = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_radius_slider = {'red': [], 'green': [], 'blue': []}
        self.color_bitplane_smoothness = {'red': [], 'green': [], 'blue': []}

        for color in ['red', 'green', 'blue']:
            for i in range(8):
                self.color_bitplane_enable[color].append(tk.BooleanVar(value=True))
                self.color_bitplane_radius[color].append(tk.IntVar(value=0))  # Default to 0
                self.color_bitplane_radius_slider[color].append(tk.DoubleVar(value=0))  # Default to 0
                self.color_bitplane_smoothness[color].append(tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS))

        self._build_ui()

        # Load saved settings BEFORE adding traces to prevent auto-switching modes
        self._load_settings()

        # Auto-expand RGB table if color_channels mode is selected
        if self.output_mode.get() == "color_channels":
            self._toggle_rgb_table()

        # Auto-expand bit plane table if grayscale_bitplanes mode is selected
        if self.output_mode.get() == "grayscale_bitplanes":
            self._toggle_bitplane_table()

        # Auto-expand color bit plane table if color_bitplanes mode is selected
        if self.output_mode.get() == "color_bitplanes":
            self._toggle_color_bitplane_table()

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
        width = 665  # Fixed width (wider for table layout)
        # Get the required height from the container
        height = self.root.winfo_reqheight()
        # Add a small buffer
        height = min(height + 20, 1000)  # Cap at 1000px
        # Position will be set by main() to center all windows
        self.root.geometry(f"{width}x{height}+0+0")
        # Set minimum window size to prevent shrinking when tables collapse
        self.root.minsize(width, 200)

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

        # Add invisible spacer frame to maintain minimum width
        spacer = ttk.Frame(container, width=630, height=1)
        spacer.pack(fill='x')
        spacer.pack_propagate(False)

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

        # Two column layout for camera selection using grid
        # Column headers
        ttk.Label(camera_frame, text="Select Camera:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(camera_frame, text="Resolution:").grid(row=0, column=1, sticky='w', padx=5, pady=2)

        # Create frame for radio buttons
        cam_buttons_frame = ttk.Frame(camera_frame)
        cam_buttons_frame.grid(row=1, column=0, sticky='nw', padx=5, pady=2)

        # Create radio buttons for each camera
        for cam in self.available_cameras:
            cam_text = f"Camera {cam['id']}"
            ttk.Radiobutton(cam_buttons_frame, text=cam_text, value=cam['id'],
                           variable=self.selected_camera,
                           command=self._on_camera_change).pack(anchor='w', pady=1)

        # Resolution dropdown - will be populated based on selected camera
        self.resolution_combobox = ttk.Combobox(camera_frame, textvariable=self.selected_resolution,
                                                 state='readonly', width=15)
        self.resolution_combobox.grid(row=1, column=1, sticky='nw', padx=5, pady=2)
        self.resolution_combobox.bind('<<ComboboxSelected>>', self._on_resolution_change)

        # Add note about performance
        ttk.Label(camera_frame, text="(smaller = faster)", font=('TkDefaultFont', 12, 'italic'),
                 foreground='gray50').grid(row=1, column=2, sticky='w', padx=(0, 5), pady=2)

        # Populate resolution dropdown for initially selected camera
        self._update_resolution_dropdown()

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
        ttk.Label(gs_right, text="Filter Cutoff Smoothness 0-100 pixels (Butterworth offset)", wraplength=250).pack(anchor='w', pady=(5, 0))
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

        # Header with radio button and expand/collapse button
        rgb_header_frame = ttk.Frame(color_channels_group)
        rgb_header_frame.pack(fill='x', pady=(0, 5))

        # Radio button on left
        ttk.Radiobutton(rgb_header_frame, text="Individual Color Channels", value="color_channels",
                       variable=self.output_mode, command=self._on_rgb_radio_select).pack(side='left')

        # Expand/collapse on right
        self.rgb_expanded = tk.BooleanVar(value=False)
        self.rgb_toggle_btn = ttk.Button(rgb_header_frame, text="▶", width=1,
                                        command=self._toggle_rgb_table)
        self.rgb_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(rgb_header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create table using grid layout (initially hidden)
        table_frame = ttk.Frame(color_channels_group)
        self.rgb_table_frame = table_frame  # Store reference for show/hide

        # Header row (row 0)
        ttk.Label(table_frame, text="").grid(row=0, column=0, padx=5, pady=2, sticky='w')  # Empty cell for color labels
        ttk.Label(table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)  # Value label column
        ttk.Label(table_frame, text="Smoothness (Butterworth offset)").grid(row=0, column=4, padx=5, pady=2)
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
        ttk.Label(bitplane_table_frame, text="Bit").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        ttk.Label(bitplane_table_frame, text="Enabled").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="Filter Radius (pixels)").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(bitplane_table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(bitplane_table_frame, text="Smoothness (Butterworth offset)").grid(row=0, column=4, padx=5, pady=2)
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

        # Row 4: Color Bit Planes - grouped section with tabbed layout
        color_bitplanes_group = ttk.LabelFrame(fft_frame, text="", padding=5)
        color_bitplanes_group.pack(fill='x', pady=(0, 10))

        # Header with radio button and expand/collapse button
        color_bp_header_frame = ttk.Frame(color_bitplanes_group)
        color_bp_header_frame.pack(fill='x', pady=(0, 5))

        # Radio button on left
        color_bp_radio = ttk.Radiobutton(color_bp_header_frame, text="Color Bit Planes",
                                        value="color_bitplanes",
                                        variable=self.output_mode,
                                        command=self._on_color_bitplane_radio_select)
        color_bp_radio.pack(side='left', anchor='w')

        # Expand/collapse on right
        self.color_bitplane_expanded = tk.BooleanVar(value=False)
        self.color_bitplane_toggle_btn = ttk.Button(color_bp_header_frame, text="▶", width=1,
                                                    command=self._toggle_color_bitplane_table)
        self.color_bitplane_toggle_btn.pack(side='right', padx=(2, 0))
        ttk.Label(color_bp_header_frame, text="Expand/Collapse").pack(side='right', padx=(5, 0))

        # Create notebook with tabs for Red, Green, Blue (initially hidden)
        color_bp_notebook_frame = ttk.Frame(color_bitplanes_group)
        self.color_bitplane_notebook_frame = color_bp_notebook_frame  # Store reference for show/hide

        # Create custom tab bar with colored labels
        tab_bar = ttk.Frame(color_bp_notebook_frame)
        tab_bar.pack(fill='x', pady=(0, 0))

        # Create a variable to track which tab is selected
        self.color_bp_selected_tab = tk.StringVar(value='red')

        # Create colored tab buttons
        tab_buttons_frame = tk.Frame(tab_bar, bg='gray85')
        tab_buttons_frame.pack(side='left', fill='x')

        # Store tab frames and buttons for switching
        self.color_bp_tab_frames = {}
        self.color_bp_tab_buttons = {}

        # Create container for tab content with border
        # Using ttk.LabelFrame with very thin border
        tab_content_container = ttk.Frame(color_bp_notebook_frame, relief='solid', borderwidth=1)
        tab_content_container.pack(fill='both', expand=True)

        # Create custom ttk styles for colored labels (before the loop to avoid shadowing ttk)
        style = ttk.Style()

        # Create tabs for Red, Green, Blue with colored labels
        # Use brighter blue for better contrast on gray background
        for color_name, color_fg in [('Red', 'red'), ('Green', 'green'), ('Blue', 'DeepSkyBlue')]:
            color_key = color_name.lower()

            # Create a custom style for colored labels in this tab
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

            # Header row (row 0) - use ttk.Label which blends better with parent frame

            ttk.Label(table_frame, text="Bit", style=style_name).grid(row=0, column=0, padx=5, pady=2, sticky='e')
            ttk.Label(table_frame, text="Enabled", style=style_name).grid(row=0, column=1, padx=5, pady=2)
            ttk.Label(table_frame, text="Filter Radius (pixels)", style=style_name).grid(row=0, column=2, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=3, padx=2, pady=2)
            ttk.Label(table_frame, text="Smoothness (Butterworth offset)", style=style_name).grid(row=0, column=4, padx=5, pady=2)
            ttk.Label(table_frame, text="").grid(row=0, column=5, padx=2, pady=2)

            # Create 8 rows for bit planes (7 MSB down to 0 LSB)
            bit_labels = ["(MSB) 7", "6", "5", "4", "3", "2", "1", "(LSB) 0"]
            color_key = color_name.lower()

            for i, label in enumerate(bit_labels):
                row = i + 1  # +1 for header row

                # Bit plane label (right-justified) - use ttk.Label with custom style
                ttk.Label(table_frame, text=label, style=style_name).grid(row=row, column=0, padx=5, pady=5, sticky='e')

                # Enabled checkbox
                ttk.Checkbutton(table_frame, variable=self.color_bitplane_enable[color_key][i]).grid(row=row, column=1, padx=5, pady=5)

                # Radius slider (exponential scale: 0-100 slider -> 0-200+ radius)
                def update_color_bp_radius(slider_val, c=color_key, idx=i):
                    radius = self._slider_to_radius(float(slider_val))
                    self.color_bitplane_radius[c][idx].set(radius)

                radius_slider = ttk.Scale(table_frame, from_=0, to=100,
                                         variable=self.color_bitplane_radius_slider[color_key][i],
                                         orient='horizontal',
                                         command=lambda v, c=color_key, idx=i: update_color_bp_radius(v, c, idx))
                radius_slider.grid(row=row, column=2, padx=5, pady=5, sticky='ew')

                # Radius value label
                tk.Label(table_frame, textvariable=self.color_bitplane_radius[color_key][i],
                        width=4, foreground=color_fg).grid(row=row, column=3, padx=(2, 10), pady=5)

                # Smoothness slider
                smooth_slider = ttk.Scale(table_frame, from_=0, to=100,
                                          variable=self.color_bitplane_smoothness[color_key][i],
                                          orient='horizontal',
                                          command=lambda v, c=color_key, idx=i: self.color_bitplane_smoothness[c][idx].set(int(float(v))))
                smooth_slider.grid(row=row, column=4, padx=5, pady=5, sticky='ew')

                # Smoothness value label
                tk.Label(table_frame, textvariable=self.color_bitplane_smoothness[color_key][i],
                        width=4, foreground=color_fg).grid(row=row, column=5, padx=2, pady=5)

            # Configure column weights for proper expansion
            table_frame.columnconfigure(2, weight=1)  # Filter Radius column expands
            table_frame.columnconfigure(4, weight=1)  # Smoothness column expands

        # Show the red tab by default
        self._switch_color_bp_tab('red')

        # Common controls at bottom
        common_frame = ttk.Frame(fft_frame)
        common_frame.pack(fill='x', pady=(10, 0))

        # Gain slider with value on right (log scale: 0.2 to 5, center=1)
        ttk.Label(common_frame, text="Gain: 0.1x to 1x (no gain) to 10x").pack(anchor='w', pady=(5, 0))
        gain_row = ttk.Frame(common_frame)
        gain_row.pack(fill='x')
        gain_slider = ttk.Scale(gain_row, from_=0.1, to=10,
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
            'color_bp_selected_tab': self.color_bp_selected_tab.get(),
            # Camera and resolution settings
            'selected_camera': self.selected_camera.get(),
            'selected_resolution': self.selected_resolution.get()
        }

        settings_file = os.path.expanduser('~/.webcam_filter_settings.json')
        try:
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            # print(f"Settings saved to {settings_file}")
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

                    # Set radius and update slider position to match
                    radius_val = color_radius[i] if i < len(color_radius) else DEFAULT_FFT_RADIUS
                    self.color_bitplane_radius[color][i].set(radius_val)
                    self.color_bitplane_radius_slider[color][i].set(self._radius_to_slider(radius_val))

                    self.color_bitplane_smoothness[color][i].set(color_smoothness[i] if i < len(color_smoothness) else DEFAULT_FFT_SMOOTHNESS)

            # Restore selected color tab if color bitplanes mode is selected
            saved_tab = settings.get('color_bp_selected_tab', 'red')
            if saved_tab in ['red', 'green', 'blue']:
                self.color_bp_selected_tab.set(saved_tab)
                # Only switch tab if color_bitplanes mode is selected
                if self.output_mode.get() == "color_bitplanes":
                    self._switch_color_bp_tab(saved_tab)

            # Restore camera and resolution settings
            saved_camera = settings.get('selected_camera')
            if saved_camera is not None:
                self.selected_camera.set(saved_camera)

            saved_resolution = settings.get('selected_resolution', '')
            if saved_resolution:
                self.selected_resolution.set(saved_resolution)

            # Update resolution dropdown for the selected camera
            self._update_resolution_dropdown()

            # print(f"Settings loaded from {settings_file}")
        except Exception as e:
            print(f"Error loading settings: {e}")

    def _on_rgb_control_change(self, *args):
        """Auto-select Individual Color Channels radio button when RGB controls are changed"""
        self.output_mode.set("color_channels")
        # Also expand the table when a control is changed
        if not self.rgb_expanded.get():
            self._toggle_rgb_table()

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

    def _toggle_rgb_table(self):
        """Toggle the visibility of the RGB channels table"""
        if self.rgb_expanded.get():
            # Collapse
            self.rgb_table_frame.pack_forget()
            self.rgb_toggle_btn.config(text="▶")
            self.rgb_expanded.set(False)
        else:
            # Expand
            self.rgb_table_frame.pack(fill='x', padx=10)
            self.rgb_toggle_btn.config(text="▼")
            self.rgb_expanded.set(True)
            # Also select the radio button when expanding
            self.output_mode.set("color_channels")

    def _on_rgb_radio_select(self):
        """Expand the RGB table when radio button is selected"""
        if not self.rgb_expanded.get():
            self._toggle_rgb_table()

    def _toggle_color_bitplane_table(self):
        """Toggle the visibility of the color bit plane notebook"""
        if self.color_bitplane_expanded.get():
            # Collapse
            self.color_bitplane_notebook_frame.pack_forget()
            self.color_bitplane_toggle_btn.config(text="▶")
            self.color_bitplane_expanded.set(False)
        else:
            # Expand
            self.color_bitplane_notebook_frame.pack(fill='both', expand=True, padx=10, pady=5)
            self.color_bitplane_toggle_btn.config(text="▼")
            self.color_bitplane_expanded.set(True)
            # Also select the radio button when expanding
            self.output_mode.set("color_bitplanes")

    def _on_color_bitplane_radio_select(self):
        """Expand the color bit plane table when radio button is selected"""
        if not self.color_bitplane_expanded.get():
            self._toggle_color_bitplane_table()

    def _switch_color_bp_tab(self, color_key):
        """Switch between Red, Green, Blue tabs in color bit plane section"""
        # Update all tab button appearances
        for key, btn in self.color_bp_tab_buttons.items():
            if key == color_key:
                # Active tab: sunken relief and white background
                btn.config(relief='sunken', bg='white')
            else:
                # Inactive tabs: raised relief and gray background
                btn.config(relief='raised', bg='gray85')

        # Hide all tab frames
        for frame in self.color_bp_tab_frames.values():
            frame.pack_forget()

        # Show the selected tab frame
        self.color_bp_tab_frames[color_key].pack(fill='both', expand=True, padx=5, pady=5)

        # Force immediate rendering to prevent progressive widget appearance
        self.root.update_idletasks()

        self.color_bp_selected_tab.set(color_key)

    def _on_camera_change(self):
        """Handle camera selection change"""
        self.camera_changed = True
        self.new_camera_id = self.selected_camera.get()
        self._update_resolution_dropdown()

    def _update_resolution_dropdown(self):
        """Update resolution dropdown based on selected camera"""
        camera_id = self.selected_camera.get()
        for cam in self.available_cameras:
            if cam['id'] == camera_id:
                resolutions = [f"{w}x{h}" for w, h in cam['resolutions']]
                self.resolution_combobox['values'] = resolutions
                # Set resolution if nothing selected yet or current selection not available
                if not self.selected_resolution.get() or self.selected_resolution.get() not in resolutions:
                    # Prefer 720p (1280x720) as default, otherwise use first available
                    default_res = "1280x720" if "1280x720" in resolutions else (resolutions[0] if resolutions else "")
                    self.selected_resolution.set(default_res)
                break

    def _on_resolution_change(self, event=None):
        """Handle resolution selection change"""
        self.resolution_changed = True
        self.new_resolution = self.selected_resolution.get()

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
    """Find all available cameras and their supported resolutions"""
    import sys
    import os

    available_cameras = []

    # Common resolutions to test (width, height)
    test_resolutions = [
        (640, 480),    # VGA
        (1280, 720),   # 720p
        (1920, 1080),  # 1080p
        (2560, 1440),  # 1440p
        (3840, 2160),  # 4K
    ]

    # Suppress OpenCV warnings during camera detection
    # Save original stderr
    original_stderr = sys.stderr
    devnull = None

    try:
        # Redirect stderr to devnull to suppress OpenCV warnings
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull

        for camera_id in range(5):  # Check first 5 camera indices
            cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    # Get default resolution
                    default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Test which resolutions are supported
                    supported_resolutions = []
                    for width, height in test_resolutions:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            # Check if we got the resolution we asked for (or close to it)
                            if abs(actual_width - width) <= 10 and abs(actual_height - height) <= 10:
                                res_str = f"{actual_width}x{actual_height}"
                                if res_str not in [f"{w}x{h}" for w, h in supported_resolutions]:
                                    supported_resolutions.append((actual_width, actual_height))

                    # If no resolutions matched, use the default
                    if not supported_resolutions:
                        supported_resolutions.append((default_width, default_height))

                    # Sort resolutions by total pixels (smallest to largest)
                    supported_resolutions.sort(key=lambda x: x[0] * x[1])

                    available_cameras.append({
                        'id': camera_id,
                        'width': default_width,
                        'height': default_height,
                        'resolutions': supported_resolutions
                    })
                    res_list = ", ".join([f"{w}x{h}" for w, h in supported_resolutions])
                    print(f"Camera {camera_id}: {res_list}")
                cap.release()
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        if devnull:
            devnull.close()

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

        # print(f"Using camera {selected_camera['id']} (from command line)")
        pass
    else:
        # Use highest numbered camera
        selected_camera = max(available_cameras, key=lambda c: c['id'])
        # print(f"Using camera {selected_camera['id']} (highest numbered camera)")
        # print(f"To use a different camera, run with: --camera <id>")

    # Initialize selected webcam
    # print(f"\nInitializing camera {selected_camera['id']}...")
    cap = cv2.VideoCapture(selected_camera['id'], cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open selected webcam")
        return

    # Create tkinter control panel first (to load resolution preference)
    # Use temporary dimensions - will be updated after applying resolution
    temp_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    temp_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    controls = ControlPanel(temp_width, temp_height, available_cameras, selected_camera['id'])

    # Apply the selected resolution from control panel
    selected_res = controls.selected_resolution.get()
    if selected_res:
        try:
            res_width, res_height = map(int, selected_res.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
        except ValueError:
            print(f"Warning: Invalid resolution format '{selected_res}', using camera default")

    # Get actual webcam dimensions after applying resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Webcam initialized: {width}x{height}")
    print("Controls:")
    print("  SPACEBAR - Toggle effect on/off")
    print("  Q, ESC, or Ctrl+C - Quit")
    print("  Use control panel to adjust parameters")

    # Initialize combined sketch filter with actual dimensions
    sketch = CombinedSketchFilter(width, height)

    # Create filter curve visualization window (after control panel so Tkinter root exists)
    sketch.fft.create_visualization_window()

    # Draw initial visualization with default parameters
    initial_radius = controls.fft_radius.get()
    initial_smoothness = controls.fft_smoothness.get()
    # Create a dummy mask and distance for initial visualization
    dummy_size = 480
    center = dummy_size // 2
    y, x = np.ogrid[:dummy_size, :dummy_size]
    dummy_distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    dummy_mask = sketch.fft._create_mask(dummy_distance, initial_radius, initial_smoothness, dummy_size, dummy_size)

    # Create Tkinter video window
    video_window = VideoWindow(controls.root, title='Sketch Filter', width=width, height=height)

    # Calculate centered window positions
    # Get screen dimensions
    screen_width = controls.root.winfo_screenwidth()
    screen_height = controls.root.winfo_screenheight()

    # Force update to get actual control panel dimensions
    controls.root.update_idletasks()
    control_width = controls.root.winfo_width()
    control_height = controls.root.winfo_height()

    # Window dimensions
    gap = 20  # Gap between windows horizontally
    vertical_gap = 100  # Preferred gap between control panel and graph (can be reduced if needed)
    video_width = width
    video_height = height
    graph_width = 600
    graph_height = 400

    # Calculate total width needed for control panel + video
    total_width = control_width + gap + video_width

    # Center the layout horizontally
    start_x = max(0, (screen_width - total_width) // 2)

    # Position control panel at top
    control_x = start_x
    control_y = 0
    controls.root.geometry(f"{control_width}x{control_height}+{control_x}+{control_y}")

    # Position video window to the right of control panel
    video_x = control_x + control_width + gap
    video_y = 0
    video_window.window.geometry(f"{video_width}x{video_height}+{video_x}+{video_y}")

    # Position graph window below control panel with fallback for screen size
    graph_x = control_x
    # Try preferred gap, but reduce if it would go off screen
    graph_y_preferred = control_y + control_height + vertical_gap

    # Check if graph fits with preferred gap
    if graph_y_preferred + graph_height <= screen_height:
        graph_y = graph_y_preferred
    else:
        # Reduce gap to fit on screen, minimum gap of 20px
        graph_y = max(control_y + control_height + gap, screen_height - graph_height - 20)

    sketch.fft.viz_window.geometry(f"{graph_width}x{graph_height}+{graph_x}+{graph_y}")

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

        # Check if resolution was changed
        if controls.resolution_changed:
            controls.resolution_changed = False
            new_resolution = controls.new_resolution
            print(f"\nChanging resolution to {new_resolution}...")

            try:
                # Parse resolution string "WIDTHxHEIGHT"
                res_width, res_height = map(int, new_resolution.split('x'))

                # Set new resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)

                # Verify the resolution was set
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if abs(actual_width - res_width) <= 10 and abs(actual_height - res_height) <= 10:
                    # Reinitialize sketch filter with new dimensions
                    sketch = CombinedSketchFilter(actual_width, actual_height)

                    # Resize video window to match new resolution
                    video_window.resize(actual_width, actual_height)

                    print(f"Resolution changed to {actual_width}x{actual_height}")
                else:
                    print(f"Warning: Requested {res_width}x{res_height}, got {actual_width}x{actual_height}")
                    # Update the dropdown to show actual resolution
                    controls.selected_resolution.set(f"{actual_width}x{actual_height}")
                    # Still reinitialize with actual dimensions
                    sketch = CombinedSketchFilter(actual_width, actual_height)

                # Skip this frame and get a fresh one with the new resolution
                continue
            except Exception as e:
                print(f"Error changing resolution: {e}")
                # Revert to current resolution
                current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                controls.selected_resolution.set(f"{current_width}x{current_height}")

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
            elif output_mode == "color_bitplanes":
                # Color Bit Planes mode - filter each bit plane of each color independently
                color_bitplane_params = {}
                for color in ['red', 'green', 'blue']:
                    bitplane_params = []
                    for i in range(8):
                        bitplane_params.append({
                            'enable': controls.color_bitplane_enable[color][i].get(),
                            'radius': controls.color_bitplane_radius[color][i].get(),
                            'smoothness': controls.color_bitplane_smoothness[color][i].get()
                        })
                    color_bitplane_params[color] = bitplane_params
                result = sketch.fft.draw_color_bitplanes(frame, color_bitplane_params)
            else:
                # Grayscale composite mode (and other modes to be implemented)
                # Update FFT filter parameters from UI controls
                sketch.fft.radius = controls.fft_radius.get()
                sketch.fft.smoothness = controls.fft_smoothness.get()
                sketch.fft.show_fft = controls.show_fft.get()
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

        # Display FPS at top left
        y_offset = 30
        cv2.putText(result, f"FPS: {fps:.1f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # line_height = 30
        # cv2.putText(result, f"FFT Radius: {sketch.fft.radius}", (10, y_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # y_offset += line_height
        # cv2.putText(result, f"FFT Smoothness: {sketch.fft.smoothness}", (10, y_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # y_offset += line_height
        # cv2.putText(result, f"Show FFT: {sketch.fft.show_fft}", (10, y_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # y_offset += line_height

        # Display effect status
        # status_text = "EFFECT ON" if effect_enabled else "EFFECT OFF (SPACEBAR to toggle)"
        # status_color = (0, 255, 0) if effect_enabled else (0, 165, 255)  # Green if on, orange if off
        # cv2.putText(result, status_text, (10, y_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

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

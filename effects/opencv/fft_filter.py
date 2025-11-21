"""
FFT High-Pass Filter effect using OpenCV.

Simple FFT-based frequency filter with spectrum visualization,
filter response graph, and difference window.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from core.base_effect import BaseUIEffect
from core.form_renderer import Subform, EffectForm
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import math
import time


# Configuration Constants
DEFAULT_FFT_RADIUS = 0
DEFAULT_FFT_SMOOTHNESS = 0
DEFAULT_SHOW_FFT = False

# Butterworth Filter Parameters
BUTTERWORTH_ORDER_MAX = 10.0
BUTTERWORTH_ORDER_MIN = 0.5
BUTTERWORTH_ORDER_RANGE = 9.5
BUTTERWORTH_SMOOTHNESS_SCALE = 100.0
BUTTERWORTH_TARGET_ATTENUATION = 0.03
BUTTERWORTH_DIVISION_EPSILON = 1e-10

# Visualization Window Settings
VIZ_WINDOW_WIDTH = 600
VIZ_WINDOW_HEIGHT = 400
VIZ_FIGURE_WIDTH_INCHES = 6
VIZ_FIGURE_HEIGHT_INCHES = 4
VIZ_FIGURE_DPI = 100
VIZ_Y_AXIS_MIN = -0.1
VIZ_Y_AXIS_MAX = 1.1
VIZ_GRID_ALPHA = 0.3


class FFTFilterEffect(BaseUIEffect):
    """Simple FFT high-pass filter with visualization"""

    def __init__(self, width, height, root=None):
        super().__init__(width, height, root)

        # Control variables
        self.enabled = tk.BooleanVar(value=True)
        self.fft_radius = tk.IntVar(value=DEFAULT_FFT_RADIUS)
        self.fft_smoothness = tk.IntVar(value=DEFAULT_FFT_SMOOTHNESS)
        self.show_fft = tk.BooleanVar(value=DEFAULT_SHOW_FFT)

        # Visualization window
        self.viz_window = None
        self.viz_fig = None
        self.viz_ax = None
        self.viz_canvas = None

        # Difference window
        self.diff_window = None
        self.diff_label = None
        self.input_frame = None
        self.diff_frame = None

        # Performance optimization: frame caching
        self.last_result = None
        self.last_frame_time = 0
        self.min_frame_interval = 0.05  # ~20 FPS max for FFT processing

    @classmethod
    def get_name(cls) -> str:
        return "FFT High-Pass Filter"

    @classmethod
    def get_description(cls) -> str:
        return "FFT-based high-pass filter with spectrum visualization"

    @classmethod
    def get_method_signature(cls) -> str:
        return "np.fft.fft2() / np.fft.ifft2()"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

    def get_form_schema(self):
        """Return the form schema for this effect's parameters"""
        return [
            {'type': 'slider', 'label': 'Radius', 'key': 'fft_radius', 'min': 0, 'max': 200, 'default': 0},
            {'type': 'slider', 'label': 'Smoothness', 'key': 'fft_smoothness', 'min': 0, 'max': 100, 'default': 0},
            {'type': 'checkbox', 'label': 'Show FFT Spectrum', 'key': 'show_fft', 'default': False},
        ]

    def get_current_data(self):
        """Get current parameter values as a dictionary"""
        return {
            'fft_radius': self.fft_radius.get(),
            'fft_smoothness': self.fft_smoothness.get(),
            'show_fft': self.show_fft.get()
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

        # Store reference to subform for syncing values back
        self._update_vars_from_subform()

        # Create visualization and difference windows
        self._create_visualization_window()
        self._update_visualization()
        self._create_diff_window()

        return self.control_panel

    def _update_vars_from_subform(self):
        """Set up tracing to sync subform values back to effect variables"""
        # When subform values change, update effect's tk.Variables
        for key, var in self._subform._vars.items():
            if key == 'fft_radius':
                var.trace_add('write', lambda *args: self._on_radius_change())
            elif key == 'fft_smoothness':
                var.trace_add('write', lambda *args: self._on_smoothness_change())
            elif key == 'show_fft':
                var.trace_add('write', lambda *args: self.show_fft.set(self._subform._vars['show_fft'].get()))

    def _on_radius_change(self):
        """Handle radius change and update visualization"""
        self.fft_radius.set(int(self._subform._vars['fft_radius'].get()))
        self._update_visualization()

    def _on_smoothness_change(self):
        """Handle smoothness change and update visualization"""
        self.fft_smoothness.set(int(self._subform._vars['fft_smoothness'].get()))
        self._update_visualization()

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
            self._update_vars_from_subform()

    def _copy_text(self):
        """Copy settings as human-readable text to clipboard"""
        lines = [self.get_name()]
        lines.append(self.get_description())
        lines.append(self.get_method_signature())
        lines.append(f"Radius: {self.fft_radius.get()}")
        lines.append(f"Smoothness: {self.fft_smoothness.get()}")
        lines.append(f"Show FFT Spectrum: {'Yes' if self.show_fft.get() else 'No'}")

        text = '\n'.join(lines)
        if self.root_window:
            self.root_window.clipboard_clear()
            self.root_window.clipboard_append(text)

    def _copy_json(self):
        """Copy settings as JSON to clipboard"""
        data = {
            'effect': self.get_name(),
            'fft_radius': self.fft_radius.get(),
            'fft_smoothness': self.fft_smoothness.get(),
            'show_fft': self.show_fft.get()
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
            lines = text.strip().split('\n')

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if 'radius' in key:
                        self.fft_radius.set(max(0, min(200, int(value))))
                    elif 'smoothness' in key:
                        self.fft_smoothness.set(max(0, min(100, int(value))))
                    elif 'show' in key and 'fft' in key:
                        self.show_fft.set(value.lower() in ('yes', 'true', '1'))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'fft_radius' in self._subform._vars:
                    self._subform._vars['fft_radius'].set(self.fft_radius.get())
                if 'fft_smoothness' in self._subform._vars:
                    self._subform._vars['fft_smoothness'].set(self.fft_smoothness.get())
                if 'show_fft' in self._subform._vars:
                    self._subform._vars['show_fft'].set(self.show_fft.get())
        except Exception as e:
            print(f"Error pasting text: {e}")

    def _paste_json(self):
        """Paste settings from JSON on clipboard"""
        if not self.root_window:
            return

        try:
            text = self.root_window.clipboard_get()
            data = json.loads(text)

            if 'fft_radius' in data:
                self.fft_radius.set(max(0, min(200, int(data['fft_radius']))))
            if 'fft_smoothness' in data:
                self.fft_smoothness.set(max(0, min(100, int(data['fft_smoothness']))))
            if 'show_fft' in data:
                self.show_fft.set(bool(data['show_fft']))

            # Update subform variables if in edit mode
            if self._current_mode == 'edit' and hasattr(self, '_subform'):
                if 'fft_radius' in self._subform._vars:
                    self._subform._vars['fft_radius'].set(self.fft_radius.get())
                if 'fft_smoothness' in self._subform._vars:
                    self._subform._vars['fft_smoothness'].set(self.fft_smoothness.get())
                if 'show_fft' in self._subform._vars:
                    self._subform._vars['show_fft'].set(self.show_fft.get())
        except Exception as e:
            print(f"Error pasting JSON: {e}")

    def get_view_mode_summary(self) -> str:
        """Return a formatted summary of current settings for view mode"""
        lines = []
        lines.append(f"Radius: {self.fft_radius.get()}")
        lines.append(f"Smoothness: {self.fft_smoothness.get()}")
        lines.append(f"Show FFT Spectrum: {'Yes' if self.show_fft.get() else 'No'}")
        return '\n'.join(lines)

    def _create_visualization_window(self):
        """Create matplotlib window to visualize the filter curve"""
        if self.viz_window is not None or self.root_window is None:
            return

        self.viz_window = tk.Toplevel(self.root_window)
        self.viz_window.withdraw()  # Hide until positioned by main.py
        self.viz_window.title("Filter Curve Visualization")

        # Position to the right of pipeline UI, below video windows
        # Get screen dimensions
        screen_width = self.viz_window.winfo_screenwidth()

        # Video windows are roughly in center-right of screen, at top
        # Place viz window below them (video height + margin) and to the right
        video_x = screen_width // 2 - self.width // 4
        viz_x = video_x + 200  # Offset further right from video window
        viz_y = self.height + 100  # Below video windows

        self.viz_window.geometry(f"{VIZ_WINDOW_WIDTH}x{VIZ_WINDOW_HEIGHT}+{viz_x}+{viz_y}")

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
            return

        self.diff_window = tk.Toplevel(self.root_window)
        self.diff_window.withdraw()  # Hide until positioned by main.py
        self.diff_window.title("Difference View (Blocked Frequencies)")

        # Create label to hold the image
        self.diff_label = tk.Label(self.diff_window, bg='black')
        self.diff_label.pack()

        # Position to the right of center screen with overlap
        # Get screen dimensions
        screen_width = self.diff_window.winfo_screenwidth()

        # Estimate video window position (centered layout)
        # Video is roughly in the right half of screen
        video_x = screen_width // 2 - self.width // 4

        # Position diff window to overlap right edge of video by ~15%
        overlap = int(self.width * 0.15)
        diff_x = video_x + self.width - overlap
        diff_y = 50

        self.diff_window.geometry(f"{self.width}x{self.height}+{diff_x}+{diff_y}")

        # Don't let closing this window close the app
        self.diff_window.protocol("WM_DELETE_WINDOW", self._close_diff_window)

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

        # Resize window if frame dimensions changed
        frame_height, frame_width = self.diff_frame.shape[:2]
        if frame_width != self.width or frame_height != self.height:
            self.width = frame_width
            self.height = frame_height
            self.diff_window.geometry(f"{frame_width}x{frame_height}")

    def _update_visualization(self):
        """Update the visualization graph based on current settings"""
        if not self.viz_ax or not self.viz_canvas:
            return

        # Clear previous plot
        self.viz_ax.clear()
        self.viz_ax.set_xlabel('Distance from Center (pixels)')
        self.viz_ax.set_ylabel('Mask Value (0=blocked, 1=passed)')
        self.viz_ax.set_title('FFT Filter Transition Curve')
        self.viz_ax.grid(True, alpha=VIZ_GRID_ALPHA)
        self.viz_ax.set_ylim(VIZ_Y_AXIS_MIN, VIZ_Y_AXIS_MAX)

        # Generate x-axis (distance from center)
        max_distance = 200
        distances = np.linspace(0, max_distance, 1000)

        # Plot single curve for grayscale
        radius = self.fft_radius.get()
        smoothness = self.fft_smoothness.get()
        curve = self._compute_filter_curve(distances, radius, smoothness)
        self.viz_ax.plot(distances, curve, 'b-', linewidth=2, label='Filter Response')

        # Add vertical line at radius
        if radius > 0:
            self.viz_ax.axvline(x=radius, color='r', linestyle='--', alpha=0.5, label=f'Radius={radius}')

        self.viz_ax.legend()

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

    def _apply_fft_to_channel(self, channel, radius, smoothness, return_fft=False, return_blocked=False, draw_circle=False):
        """Apply FFT filter to a single channel

        Args:
            channel: Input channel
            radius: Filter radius
            smoothness: Filter smoothness
            return_fft: If True, return FFT magnitude spectrum instead of filtered result
            return_blocked: If True, return FFT magnitude spectrum of BLOCKED frequencies
            draw_circle: If True and return_fft/return_blocked, draw circle on grayscale spectrum
        """
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

        # If showing FFT, return the magnitude spectrum
        if return_fft:
            # Apply mask to show what passes through
            fshift_masked = dft_shift * mask
            magnitude = cv2.magnitude(fshift_masked[:, :, 0], fshift_masked[:, :, 1])
            magnitude = np.log(magnitude + 1)  # Log scale for better visualization
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_uint8 = magnitude.astype(np.uint8)

            if draw_circle:
                # Convert to BGR for circle drawing
                magnitude_bgr = cv2.cvtColor(magnitude_uint8, cv2.COLOR_GRAY2BGR)
                # Draw circle showing the filter radius
                if radius > 0:
                    cv2.circle(magnitude_bgr, (ccol, crow), int(radius), (0, 255, 0), 2)
                return magnitude_bgr
            else:
                return magnitude_uint8

        # If showing blocked frequencies, return spectrum of what was removed
        if return_blocked:
            # Invert the mask to get what was blocked
            inverse_mask = np.ones_like(mask) - mask
            blocked_fshift = dft_shift * inverse_mask
            magnitude = cv2.magnitude(blocked_fshift[:, :, 0], blocked_fshift[:, :, 1])
            magnitude = np.log(magnitude + 1)  # Log scale for better visualization
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_uint8 = magnitude.astype(np.uint8)

            if draw_circle:
                # Convert to BGR for circle drawing
                magnitude_bgr = cv2.cvtColor(magnitude_uint8, cv2.COLOR_GRAY2BGR)
                # Draw circle showing the filter radius
                if radius > 0:
                    cv2.circle(magnitude_bgr, (ccol, crow), int(radius), (0, 255, 0), 2)
                return magnitude_bgr
            else:
                return magnitude_uint8

        # Apply mask
        fshift = dft_shift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Clip to 0-255 range (don't normalize - preserves original values when radius=0)
        high_pass = np.clip(img_back, 0, 255).astype(np.uint8)

        return high_pass

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Apply FFT-based high-pass filter to the frame"""
        if not self.enabled.get():
            return frame

        # Performance optimization: Skip processing if we're going too fast
        current_time = time.time()
        time_since_last_frame = current_time - self.last_frame_time

        # If we processed a frame recently, reuse last result
        if self.last_result is not None and time_since_last_frame < self.min_frame_interval:
            # Check if frame size changed
            if self.last_result.shape == frame.shape:
                return self.last_result.copy()

        self.last_frame_time = current_time

        # Store input frame for difference calculation
        self.input_frame = frame.copy()

        radius = self.fft_radius.get()
        smoothness = self.fft_smoothness.get()
        show_fft = self.show_fft.get()

        # Split into BGR channels
        b, g, r = cv2.split(frame)

        # Apply FFT filter to each channel
        b_filtered = self._apply_fft_to_channel(b, radius, smoothness, return_fft=show_fft, draw_circle=False)
        g_filtered = self._apply_fft_to_channel(g, radius, smoothness, return_fft=show_fft, draw_circle=False)
        r_filtered = self._apply_fft_to_channel(r, radius, smoothness, return_fft=show_fft, draw_circle=False)

        # Merge filtered channels
        result = cv2.merge([b_filtered, g_filtered, r_filtered])

        # Calculate difference frame
        if show_fft:
            # When showing FFT, diff shows the blocked spectrum
            b_blocked = self._apply_fft_to_channel(b, radius, smoothness, return_blocked=True, draw_circle=False)
            g_blocked = self._apply_fft_to_channel(g, radius, smoothness, return_blocked=True, draw_circle=False)
            r_blocked = self._apply_fft_to_channel(r, radius, smoothness, return_blocked=True, draw_circle=False)
            self.diff_frame = cv2.merge([b_blocked, g_blocked, r_blocked])

            # Draw circle on both result and diff
            if radius > 0:
                rows, cols = b.shape
                crow, ccol = rows // 2, cols // 2
                cv2.circle(result, (ccol, crow), int(radius), (0, 255, 0), 2)
                cv2.circle(self.diff_frame, (ccol, crow), int(radius), (0, 255, 0), 2)
        else:
            # Normal mode: show image difference
            # Threshold to ignore sub-pixel FFT rounding errors (values of 0-1)
            diff = cv2.absdiff(self.input_frame, result)
            _, self.diff_frame = cv2.threshold(diff, 1, 255, cv2.THRESH_TOZERO)

        # Update difference window display
        self._update_diff_window()

        # Cache the result for frame skipping
        self.last_result = result.copy()

        return result

    def cleanup(self):
        """Clean up resources"""
        self._close_visualization_window()
        self._close_diff_window()

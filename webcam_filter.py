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


# Configuration constants - Canny Edge Detection
DEFAULT_CANNY_BLUR_KERNEL = 3  # Default blur kernel size for Canny (must be odd)
DEFAULT_THRESHOLD1 = 25  # Default Canny lower threshold
DEFAULT_THRESHOLD2 = 7   # Default Canny upper threshold
DEFAULT_APERTURE_SIZE = 3  # Default Sobel kernel size (3, 5, or 7)
DEFAULT_L2_GRADIENT = True  # Default gradient calculation method (True = L2, False = L1)

# Configuration constants - High-Pass Filter
DEFAULT_FREQUENCY_BLUR_KERNEL = 95  # Default blur kernel size for high-pass filter (must be odd)
DEFAULT_FREQUENCY_BOOST = 2.5  # Default frequency boost multiplier (-5 to +5, 0 = no boost)

# Configuration constants - Combined
DEFAULT_APPLY_CANNY = True  # Default: apply Canny edge detection
DEFAULT_APPLY_FREQUENCY = True  # Default: apply frequency filter
DEFAULT_INVERT = True  # Default: invert the final image


class CannyEdgeDetector:
    """Simple Canny edge detector with adjustable parameters"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Canny parameters with custom defaults
        self.blur_kernel = DEFAULT_CANNY_BLUR_KERNEL
        self.threshold1 = DEFAULT_THRESHOLD1
        self.threshold2 = DEFAULT_THRESHOLD2
        self.aperture_size = DEFAULT_APERTURE_SIZE
        self.l2_gradient = DEFAULT_L2_GRADIENT

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply Canny edge detection and return edges as grayscale image"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur (only if blur_kernel > 1)
        if self.blur_kernel > 1:
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        else:
            blurred = gray

        # Apply Canny edge detection with all parameters
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2,
                         apertureSize=self.aperture_size, L2gradient=self.l2_gradient)

        return edges


class HighPassFilter:
    """High-pass filter to extract fine details from image"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Filter parameter - kernel size for low-pass filter (must be odd)
        self.blur_kernel = DEFAULT_FREQUENCY_BLUR_KERNEL  # Larger kernel = more high-frequency details
        self.boost = DEFAULT_FREQUENCY_BOOST  # Boost multiplier for frequency filter

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply high-pass filter to extract fine details"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create low-pass version by blurring
        if self.blur_kernel > 1:
            low_pass = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        else:
            low_pass = gray

        # High-pass = abs(original - low-pass)
        # Use absolute difference so no signal = 0 (black background)
        high_pass = cv2.absdiff(gray, low_pass)

        # Apply boost if set
        if self.boost != 0:
            # Convert to float for proper multiplication
            high_pass_float = high_pass.astype(np.float32)
            # Apply boost multiplier (1 + boost means 0 = no change, positive = amplify, negative = reduce)
            high_pass_float = high_pass_float * (1.0 + self.boost)
            # Clip to valid range and convert back
            high_pass = np.clip(high_pass_float, 0, 255).astype(np.uint8)

        return high_pass


class ControlPanel:
    """Tkinter GUI control panel for filter parameters"""

    def __init__(self, width, height, available_cameras, selected_camera_id):
        self.width = width
        self.height = height
        self.max_dimension = max(width, height)
        self.available_cameras = available_cameras
        self.camera_changed = False
        self.new_camera_id = None

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Sketch Filter Controls")
        self.root.geometry("400x780")
        self.root.attributes('-topmost', True)  # Keep window on top

        # Camera selection variable
        self.selected_camera = tk.IntVar(value=selected_camera_id)

        # Variables for controls
        self.apply_frequency = tk.BooleanVar(value=DEFAULT_APPLY_FREQUENCY)
        self.frequency_blur = tk.IntVar(value=DEFAULT_FREQUENCY_BLUR_KERNEL)
        self.frequency_boost = tk.DoubleVar(value=DEFAULT_FREQUENCY_BOOST)

        self.apply_canny = tk.BooleanVar(value=DEFAULT_APPLY_CANNY)
        self.canny_blur = tk.IntVar(value=DEFAULT_CANNY_BLUR_KERNEL)
        self.threshold1 = tk.IntVar(value=DEFAULT_THRESHOLD1)
        self.threshold2 = tk.IntVar(value=DEFAULT_THRESHOLD2)
        self.aperture = tk.IntVar(value=DEFAULT_APERTURE_SIZE)
        self.l2_gradient = tk.BooleanVar(value=DEFAULT_L2_GRADIENT)

        self.invert = tk.BooleanVar(value=DEFAULT_INVERT)

        self._build_ui()

        # Flag to check if window is still open
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        """Build the tkinter UI"""
        padding = {'padx': 10, 'pady': 5}

        # Camera Selection Section
        camera_frame = ttk.LabelFrame(self.root, text="Camera Selection", padding=10)
        camera_frame.pack(fill='x', **padding)

        ttk.Label(camera_frame, text="Select Camera:").pack(anchor='w')

        # Create radio buttons for each camera
        for cam in self.available_cameras:
            cam_text = f"Camera {cam['id']}: {cam['width']}x{cam['height']}"
            ttk.Radiobutton(camera_frame, text=cam_text, value=cam['id'],
                           variable=self.selected_camera,
                           command=self._on_camera_change).pack(anchor='w', pady=2)

        # Frequency Filter Section
        freq_frame = ttk.LabelFrame(self.root, text="Frequency Filter", padding=10)
        freq_frame.pack(fill='x', **padding)

        ttk.Checkbutton(freq_frame, text="Apply Frequency Filter",
                       variable=self.apply_frequency).pack(anchor='w')

        ttk.Label(freq_frame, text=f"Blur Kernel (1-{self.max_dimension}, odd)").pack(anchor='w')
        freq_slider = ttk.Scale(freq_frame, from_=1, to=self.max_dimension,
                               variable=self.frequency_blur, orient='horizontal',
                               command=lambda v: self.frequency_blur.set(int(float(v))))
        freq_slider.pack(fill='x')
        ttk.Label(freq_frame, textvariable=self.frequency_blur).pack(anchor='e')

        ttk.Label(freq_frame, text="Boost (-5x to +5x)").pack(anchor='w', pady=(10, 0))
        boost_slider = ttk.Scale(freq_frame, from_=-5, to=5,
                                variable=self.frequency_boost, orient='horizontal')
        boost_slider.pack(fill='x')
        ttk.Label(freq_frame, textvariable=self.frequency_boost).pack(anchor='e')

        # Canny Edge Detection Section
        canny_frame = ttk.LabelFrame(self.root, text="Canny Edge Detection", padding=10)
        canny_frame.pack(fill='x', **padding)

        ttk.Checkbutton(canny_frame, text="Apply Canny Edge Detection",
                       variable=self.apply_canny).pack(anchor='w')

        ttk.Label(canny_frame, text="Blur Kernel (1-31, odd)").pack(anchor='w')
        canny_blur_slider = ttk.Scale(canny_frame, from_=1, to=31,
                                      variable=self.canny_blur, orient='horizontal',
                                      command=lambda v: self.canny_blur.set(int(float(v))))
        canny_blur_slider.pack(fill='x')
        ttk.Label(canny_frame, textvariable=self.canny_blur).pack(anchor='e')

        ttk.Label(canny_frame, text="Threshold 1 (0-255)").pack(anchor='w')
        thresh1_slider = ttk.Scale(canny_frame, from_=0, to=255,
                                   variable=self.threshold1, orient='horizontal',
                                   command=lambda v: self.threshold1.set(int(float(v))))
        thresh1_slider.pack(fill='x')
        ttk.Label(canny_frame, textvariable=self.threshold1).pack(anchor='e')

        ttk.Label(canny_frame, text="Threshold 2 (0-255)").pack(anchor='w')
        thresh2_slider = ttk.Scale(canny_frame, from_=0, to=255,
                                   variable=self.threshold2, orient='horizontal',
                                   command=lambda v: self.threshold2.set(int(float(v))))
        thresh2_slider.pack(fill='x')
        ttk.Label(canny_frame, textvariable=self.threshold2).pack(anchor='e')

        ttk.Label(canny_frame, text="Aperture Size").pack(anchor='w')
        aperture_frame = ttk.Frame(canny_frame)
        aperture_frame.pack(fill='x')
        for val in [3, 5, 7]:
            ttk.Radiobutton(aperture_frame, text=str(val), value=val,
                           variable=self.aperture).pack(side='left', padx=5)

        ttk.Checkbutton(canny_frame, text="L2 Gradient",
                       variable=self.l2_gradient).pack(anchor='w', pady=5)

        # Output Section
        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill='x', **padding)

        ttk.Checkbutton(output_frame, text="Invert (Black lines on White)",
                       variable=self.invert).pack(anchor='w', pady=2)

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

    def ensure_odd(self, value):
        """Ensure value is odd"""
        val = int(value)
        if val % 2 == 0:
            val += 1
        return val

    def get_frequency_blur(self):
        """Get frequency blur kernel (guaranteed odd)"""
        return self.ensure_odd(self.frequency_blur.get())

    def get_canny_blur(self):
        """Get canny blur kernel (guaranteed odd)"""
        return self.ensure_odd(self.canny_blur.get())


class CombinedSketchFilter:
    """Combines Canny edge detection and frequency filter"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.canny = CannyEdgeDetector(width, height)
        self.frequency = HighPassFilter(width, height)

        self.apply_canny = DEFAULT_APPLY_CANNY
        self.apply_frequency = DEFAULT_APPLY_FREQUENCY
        self.invert = DEFAULT_INVERT

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply combined filters and return result"""
        result = np.zeros((self.height, self.width), dtype=np.uint8)

        # Apply Canny if enabled
        if self.apply_canny:
            canny_result = self.canny.draw(frame)
            result = cv2.add(result, canny_result)

        # Apply frequency filter if enabled
        if self.apply_frequency:
            frequency_result = self.frequency.draw(frame)
            result = cv2.add(result, frequency_result)

        # Invert if enabled
        if self.invert:
            result = cv2.bitwise_not(result)

        # Convert to 3-channel for display
        result_3channel = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

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
    parser = argparse.ArgumentParser(description='Sketch Filter with Canny and Frequency filters')
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

    # Mode toggle
    effect_enabled = True  # Start with effect ON

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

                # Update control panel max dimension for frequency filter
                controls.max_dimension = max(new_width, new_height)

                print(f"Now using camera {selected_camera['id']} at {new_width}x{new_height}")

                # Skip this frame and get a fresh one from the new camera
                continue

        # Mirror the image (flip horizontally)
        frame = cv2.flip(frame, 1)

        # Read values from tkinter controls and update parameters
        sketch.apply_frequency = controls.apply_frequency.get()
        sketch.frequency.blur_kernel = controls.get_frequency_blur()
        sketch.frequency.boost = controls.frequency_boost.get()

        sketch.apply_canny = controls.apply_canny.get()
        sketch.canny.blur_kernel = controls.get_canny_blur()
        sketch.canny.threshold1 = controls.threshold1.get()
        sketch.canny.threshold2 = controls.threshold2.get()
        sketch.canny.aperture_size = controls.aperture.get()
        sketch.canny.l2_gradient = controls.l2_gradient.get()

        sketch.invert = controls.invert.get()

        if effect_enabled:
            # Sketch filter mode
            sketch.update()
            result = sketch.draw(frame)
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

        cv2.putText(result, f"Apply Frequency: {sketch.apply_frequency}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Frequency Blur: {sketch.frequency.blur_kernel}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Frequency Boost: {sketch.frequency.boost:.1f}x", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        cv2.putText(result, f"Apply Canny: {sketch.apply_canny}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Canny Blur: {sketch.canny.blur_kernel}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Canny Thresh1: {sketch.canny.threshold1}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Canny Thresh2: {sketch.canny.threshold2}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Canny Aperture: {sketch.canny.aperture_size}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        cv2.putText(result, f"Canny L2Grad: {sketch.canny.l2_gradient}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height

        cv2.putText(result, f"Invert: {sketch.invert}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
            elif key == ord(' '):  # Spacebar
                effect_enabled = not effect_enabled
                if effect_enabled:
                    print("Effect enabled!")
                else:
                    print("Effect disabled - showing raw webcam")
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

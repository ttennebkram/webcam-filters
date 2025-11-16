#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
import os
import signal


# Configuration constants
DEFAULT_BLUR_KERNEL = 95  # Default blur kernel size for high-pass filter (must be odd)


class HighPassFilter:
    """High-pass filter to extract fine details from image"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Filter parameter - kernel size for low-pass filter (must be odd)
        self.blur_kernel = DEFAULT_BLUR_KERNEL  # Larger kernel = more high-frequency details

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

        # Convert back to 3-channel for display
        result = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)

        return result


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

    # Let user select camera
    print(f"\nFound {len(available_cameras)} camera(s)")

    if len(available_cameras) == 1:
        selected_camera = available_cameras[0]
        print(f"Using camera {selected_camera['id']}")
    else:
        print("\nAvailable cameras:")
        for i, cam in enumerate(available_cameras):
            print(f"  {i}. Camera {cam['id']} - {cam['width']}x{cam['height']}")

        while True:
            try:
                choice = input(f"\nSelect camera (0-{len(available_cameras)-1}): ")
                choice_idx = int(choice)
                if 0 <= choice_idx < len(available_cameras):
                    selected_camera = available_cameras[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 0 and {len(available_cameras)-1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                return

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
    print("  Use trackbar to adjust high-pass filter")

    # Start window thread for better event handling and native controls
    cv2.startWindowThread()

    # Initialize high-pass filter
    hp_filter = HighPassFilter(width, height)

    # Create main window for video display
    window_name = 'High-Pass Filter'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size
    cv2.resizeWindow(window_name, width, height)

    # Create separate controls window
    controls_window = 'Controls'
    cv2.namedWindow(controls_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_window, 500, 100)

    # Create trackbar for filter parameter
    def nothing(x):
        pass

    # Calculate max blur kernel based on longest dimension
    max_dimension = max(width, height)
    max_slider = max_dimension // 2  # Slider goes from 0 to max_dimension/2

    # Blur kernel size (slider maps to odd values: 1,3,5,7,...)
    default_slider_pos = (DEFAULT_BLUR_KERNEL - 1) // 2  # Convert kernel size to slider position
    cv2.createTrackbar(f'Blur Kernel (1-{max_dimension})', controls_window, default_slider_pos, max_slider, nothing)

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

        # Mirror the image (flip horizontally)
        frame = cv2.flip(frame, 1)

        # Read trackbar value and update filter parameter
        blur_slider = cv2.getTrackbarPos(f'Blur Kernel (1-{max_dimension})', controls_window)
        hp_filter.blur_kernel = blur_slider * 2 + 1  # Convert slider to odd values: 1,3,5,7,...

        if effect_enabled:
            # High-pass filter mode
            hp_filter.update()
            result = hp_filter.draw(frame)
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

        # Display FPS and filter parameter at top left
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Blur Kernel: {hp_filter.blur_kernel}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the result
        cv2.imshow(window_name, result)

        # Show empty image in controls window (just for trackbar)
        cv2.imshow(controls_window, np.zeros((1, 500, 3), dtype=np.uint8))

        # Check if either window was closed via close button
        if (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or
            cv2.getWindowProperty(controls_window, cv2.WND_PROP_VISIBLE) < 1):
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

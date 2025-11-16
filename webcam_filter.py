#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
import os
import signal


class CannyEdgeDetector:
    """Simple Canny edge detector with adjustable parameters"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Canny parameters with custom defaults
        self.blur_kernel = 3  # Default: 3 (3x3 blur)
        self.threshold1 = 25  # Default: 25
        self.threshold2 = 7  # Default: 7
        self.aperture_size = 3  # Default: 3 (Sobel kernel size)
        self.l2_gradient = True  # Default: True (use L2 norm)

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

        # Convert to 3-channel for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

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
    print("  Use trackbars to adjust Canny parameters")

    # Start window thread for better event handling and native controls
    cv2.startWindowThread()

    # Initialize Canny edge detector
    canny = CannyEdgeDetector(width, height)

    # Create main window for video display
    window_name = 'Canny Edge Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size
    cv2.resizeWindow(window_name, width, height)

    # Create separate controls window
    controls_window = 'Controls'
    cv2.namedWindow(controls_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls_window, 500, 200)

    # Create trackbars for Canny parameters in controls window
    # Trackbar callback (does nothing, we read values in the loop)
    def nothing(x):
        pass

    # Blur kernel - any odd integer from 1 to 31 (slider 0-15 maps to 1,3,5,...,31)
    cv2.createTrackbar('Blur (1,3,5...31)', controls_window, 1, 15, nothing)  # Default: 1 -> 3
    # Threshold 1 (lower threshold)
    cv2.createTrackbar('Threshold1 (0-255)', controls_window, 25, 255, nothing)  # Default: 25
    # Threshold 2 (upper threshold)
    cv2.createTrackbar('Threshold2 (0-255)', controls_window, 7, 255, nothing)  # Default: 7
    # Aperture size - must be 3, 5, or 7 (slider 0-2 maps to exactly 3, 5, 7)
    cv2.createTrackbar('Aperture (3/5/7)', controls_window, 0, 2, nothing)  # Default: 0 -> 3
    # L2 gradient - checkbox simulation (0=Off, 1=On)
    cv2.createTrackbar('L2Grad (0=Off 1=On)', controls_window, 1, 1, nothing)  # Default: 1 (True)

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

        # Read trackbar values from controls window and update canny parameters
        blur_slider = cv2.getTrackbarPos('Blur (1,3,5...31)', controls_window)
        canny.blur_kernel = blur_slider * 2 + 1  # Convert 0-15 to 1,3,5,7,...,31
        canny.threshold1 = cv2.getTrackbarPos('Threshold1 (0-255)', controls_window)
        canny.threshold2 = cv2.getTrackbarPos('Threshold2 (0-255)', controls_window)

        aperture_slider = cv2.getTrackbarPos('Aperture (3/5/7)', controls_window)
        # Map 0->3, 1->5, 2->7 (exactly 3 options)
        aperture_map = [3, 5, 7]
        canny.aperture_size = aperture_map[aperture_slider]

        canny.l2_gradient = bool(cv2.getTrackbarPos('L2Grad (0=Off 1=On)', controls_window))

        if effect_enabled:
            # Canny edge detection mode
            canny.update()
            result = canny.draw(frame)
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
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Blur: {canny.blur_kernel}x{canny.blur_kernel}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Threshold1: {canny.threshold1}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Threshold2: {canny.threshold2}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Aperture: {canny.aperture_size}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"L2Grad: {canny.l2_gradient}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the result
        cv2.imshow(window_name, result)

        # Show empty image in controls window (just for trackbars)
        cv2.imshow(controls_window, np.zeros((1, 400, 3), dtype=np.uint8))

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

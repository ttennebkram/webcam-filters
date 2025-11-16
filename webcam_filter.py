#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import random
import time
import sys
import os
import signal


class MatrixRain:
    """Matrix-style fixed character grid with brightness-based rendering"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Falling leaves
        self.leaves = []
        for _ in range(15):  # 15 leaves falling at various times
            self.leaves.append({
                'x': random.uniform(0, width),
                'y': random.uniform(-height, 0),  # Start above screen
                'speed': random.uniform(4.5, 10.5),  # Vertical speed (3x faster)
                'oscillation_speed': random.uniform(0.02, 0.05),  # Horizontal oscillation rate
                'oscillation_amplitude': random.uniform(20, 50),  # How far left/right
                'oscillation_phase': random.uniform(0, 2 * np.pi),  # Starting phase
                'rotation': random.uniform(0, 360),
                'rotation_speed': random.uniform(-3, 3),  # Rotation per frame
                'size': random.uniform(15, 30),  # Leaf size
                'color': random.choice([
                    [0, 140, 255],   # Dark Orange
                    [0, 69, 255],    # Orange Red
                    [0, 0, 200],     # Dark Red
                    [0, 215, 255],   # Gold
                    [0, 255, 255],   # Yellow
                    [34, 139, 34],   # Forest Green
                ])
            })

    def update(self):
        """Update falling leaves"""
        for leaf in self.leaves:
            # Move down
            leaf['y'] += leaf['speed']

            # Oscillate left and right (sine wave)
            leaf['oscillation_phase'] += leaf['oscillation_speed']
            leaf['x'] += np.sin(leaf['oscillation_phase']) * leaf['oscillation_amplitude'] * 0.05

            # Rotate
            leaf['rotation'] += leaf['rotation_speed']

            # Reset if off screen
            if leaf['y'] > self.height + 50:
                leaf['y'] = random.uniform(-100, -50)
                leaf['x'] = random.uniform(0, self.width)
                leaf['speed'] = random.uniform(4.5, 10.5)  # 3x faster
                leaf['oscillation_phase'] = random.uniform(0, 2 * np.pi)

    def draw(self, frame, face_mask=None):
        """Fall leaf effect - map colors to autumn palette"""
        # Downsample to 33% for 9x speed improvement
        original_height, original_width = frame.shape[:2]
        small_frame = cv2.resize(frame, (original_width // 3, original_height // 3), interpolation=cv2.INTER_LINEAR)

        # Define fall leaf color palette (BGR format)
        fall_colors = np.array([
            [0, 100, 0],        # Dark Green
            [34, 139, 34],      # Forest Green
            [50, 205, 50],      # Lime Green
            [0, 128, 128],      # Olive Green
            [47, 107, 85],      # Dark Olive Green
            [0, 140, 255],      # Dark Orange
            [0, 69, 255],       # Orange Red
            [0, 0, 200],        # Dark Red
            [0, 0, 139],        # Maroon
            [0, 165, 255],      # Orange
            [0, 215, 255],      # Gold
            [0, 255, 255],      # Yellow
            [32, 165, 218],     # Golden Rod
            [139, 0, 139],      # Dark Magenta (for purples)
            [128, 0, 128],      # Purple
        ], dtype=np.float32)

        # Convert frame to HSV to identify blue pixels
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Create mask for blue pixels (hue ~90-130 for cyan/blue range)
        blue_mask = ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130)).astype(np.uint8)

        # Apply bilateral filter to smooth colors while preserving edges
        smooth = cv2.bilateralFilter(small_frame, 9, 75, 75)

        # For non-blue pixels, map to fall colors
        # Reshape for processing
        pixels = smooth.reshape((-1, 3)).astype(np.float32)

        # Find closest fall color for each pixel
        result_pixels = np.zeros_like(pixels)
        for i in range(len(pixels)):
            # Calculate distance to each fall color
            distances = np.linalg.norm(fall_colors - pixels[i], axis=1)
            # Assign closest fall color
            result_pixels[i] = fall_colors[np.argmin(distances)]

        result = result_pixels.astype(np.uint8).reshape((small_frame.shape))

        # Restore blue pixels from original
        blue_mask_3channel = cv2.merge([blue_mask, blue_mask, blue_mask])
        result = np.where(blue_mask_3channel > 0, small_frame, result)

        # Apply median filter to create smoother, more uniform "glass pieces"
        result = cv2.medianBlur(result, 5)

        # Optional: Add dark edges between segments for leading effect
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Create black lines for leading
        result[edges > 0] = [0, 0, 0]

        # Upscale back to original size using NEAREST to maintain blocky stained glass look
        result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Draw falling leaves
        for leaf in self.leaves:
            x = int(leaf['x'])
            y = int(leaf['y'])

            if -50 <= x < self.width + 50 and -50 <= y < self.height + 50:
                # Create a realistic leaf shape with pointed tip
                size = int(leaf['size'])

                # Create leaf shape points - teardrop/pointed leaf shape
                leaf_points = []

                # Top pointed tip
                leaf_points.append((0, -size * 0.8))

                # Right side - curved
                leaf_points.append((size * 0.3, -size * 0.4))
                leaf_points.append((size * 0.5, 0))
                leaf_points.append((size * 0.3, size * 0.4))

                # Bottom rounded
                leaf_points.append((0, size * 0.5))

                # Left side - curved
                leaf_points.append((-size * 0.3, size * 0.4))
                leaf_points.append((-size * 0.5, 0))
                leaf_points.append((-size * 0.3, -size * 0.4))

                # Rotate leaf
                rot_rad = np.radians(leaf['rotation'])
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)

                rotated_points = []
                for px, py in leaf_points:
                    rx = px * cos_r - py * sin_r
                    ry = px * sin_r + py * cos_r
                    rotated_points.append((int(x + rx), int(y + ry)))

                # Draw filled polygon for leaf
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], tuple(int(c) for c in leaf['color']), cv2.LINE_AA)

                # Add stem (small line from center)
                stem_angle = rot_rad + np.pi / 2
                stem_length = size * 0.3
                stem_end_x = int(x + stem_length * np.cos(stem_angle))
                stem_end_y = int(y + stem_length * np.sin(stem_angle))
                cv2.line(result, (x, y), (stem_end_x, stem_end_y), (0, 100, 0), 2, cv2.LINE_AA)

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

    # Start window thread for better event handling and native controls
    cv2.startWindowThread()

    # Initialize Matrix rain and edge detector
    matrix = MatrixRain(width, height)
    edge_detector = EdgeDetector()

    # Create window with native macOS controls (red/yellow/green buttons)
    window_name = 'Matrix Vision'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set initial size
    cv2.resizeWindow(window_name, width, height)

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

        if effect_enabled:
            # Matrix mode: update grid and draw based on brightness
            matrix.update()
            result = matrix.draw(frame)
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

        # Display mode and FPS
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the result
        cv2.imshow(window_name, result)

        # Check if window was closed via close button
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
    edge_detector.face_detection.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C detected - force exiting...")
        cv2.destroyAllWindows()
        os._exit(0)  # Force immediate exit
    finally:
        cv2.destroyAllWindows()

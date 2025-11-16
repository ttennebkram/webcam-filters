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
    """Winter effect with snow"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Snowflakes - irregular shapes
        self.snowflakes = []
        for _ in range(300):  # Heavy snow
            # Create small irregular shape for each snowflake
            num_points = random.randint(3, 5)
            base_size = random.uniform(1.5, 4.0)
            points = []
            for i in range(num_points):
                angle = (i / num_points) * 2 * np.pi
                radius = base_size * random.uniform(0.6, 1.0)
                px = radius * np.cos(angle)
                py = radius * np.sin(angle)
                points.append((px, py))

            self.snowflakes.append({
                'x': random.uniform(0, width),
                'y': random.uniform(-height, 0),  # Start above screen
                'speed': random.uniform(2.0, 6.0),
                'points': points,  # Irregular shape
                'drift': random.uniform(-0.5, 0.5),  # Horizontal drift
                'rotation': random.uniform(0, 360)
            })

    def update(self):
        """Update snowflakes"""
        for flake in self.snowflakes:
            # Move down
            flake['y'] += flake['speed']
            flake['x'] += flake['drift']
            flake['rotation'] += random.uniform(-2, 2)  # Slight rotation

            # Reset if off screen
            if flake['y'] > self.height:
                flake['y'] = random.uniform(-20, 0)
                flake['x'] = random.uniform(0, self.width)
                flake['speed'] = random.uniform(2.0, 6.0)
                flake['drift'] = random.uniform(-0.5, 0.5)

                # Regenerate irregular shape
                num_points = random.randint(3, 5)
                base_size = random.uniform(1.5, 4.0)
                points = []
                for i in range(num_points):
                    angle = (i / num_points) * 2 * np.pi
                    radius = base_size * random.uniform(0.6, 1.0)
                    px = radius * np.cos(angle)
                    py = radius * np.sin(angle)
                    points.append((px, py))
                flake['points'] = points

    def draw(self, frame, face_mask=None):
        """Winter effect - arctic blue near edges, white in center"""
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Use only Gaussian blurs for completely smooth, non-blocky edges
        # Start with the raw edges
        edges_processed = edges.copy().astype(np.float32)

        # Progressive blur cycles - no dilation to avoid blockiness
        # Each cycle extends reach smoothly with increasingly large blurs for fuzzier edges
        blur_sizes = [
            (15, 15), (21, 21), (27, 27), (33, 33), (39, 39), (45, 45),
            (51, 51), (57, 57), (63, 63), (71, 71), (81, 81), (91, 91)
        ]

        for blur_size in blur_sizes:
            edges_processed = cv2.GaussianBlur(edges_processed, blur_size, 0)
            # Boost opacity slightly after each blur to maintain strength
            edges_processed = np.clip(edges_processed * 1.15, 0, 255)

        # Add additional blur to feather edges more, then boost center back to 100%
        edges_blurred = cv2.GaussianBlur(edges_processed, (151, 151), 0)
        # Boost to restore center brightness while keeping feathered edges
        edges_blurred = np.clip(edges_blurred * 1.3, 0, 255).astype(np.uint8)

        # Convert frame to HSV to preserve brightness while changing color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Invert the mask - high values where there are NO edges, low values near edges
        inverted_mask = 255 - edges_blurred
        inverted_mask_normalized = inverted_mask.astype(np.float32) / 255.0

        # Apply power curve to extend arctic blue further - higher power = more arctic blue
        inverted_mask_normalized = np.power(inverted_mask_normalized, 5.0)  # 5th power for much more arctic blue

        # Near edges (inverted_mask low): arctic blue (hue 105)
        # Away from edges (inverted_mask high): white (saturation 0)

        # Blend hue: arctic blue everywhere
        hsv[:, :, 0] = 105  # Arctic blue hue

        # Blend saturation: high near edges (arctic blue), low away from edges (white)
        hsv[:, :, 1] = 255 * (1.0 - inverted_mask_normalized)  # Low saturation = white, max saturation near edges

        # Keep original brightness (value channel)
        # hsv[:, :, 2] stays unchanged

        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Now add white glow right at the edges on top
        # Use the original edge mask with less blur for tighter white edges
        white_edges = cv2.GaussianBlur(edges_processed, (31, 31), 0)
        white_edges = np.clip(white_edges * 1.2, 0, 255).astype(np.uint8)

        # Convert to 3-channel for blending
        white_edges_3channel = cv2.merge([white_edges, white_edges, white_edges])

        # Create white color
        white = np.ones_like(result, dtype=np.uint8) * 255

        # Blend white on top of arctic blue at edges
        alpha = white_edges_3channel.astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - alpha) + white.astype(np.float32) * alpha).astype(np.uint8)

        # Draw snowflakes - irregular white blobs
        for flake in self.snowflakes:
            x = int(flake['x'])
            y = int(flake['y'])

            if -10 <= x < self.width + 10 and -10 <= y < self.height + 10:
                # Rotate and position the irregular shape points
                rot_rad = np.radians(flake['rotation'])
                cos_r = np.cos(rot_rad)
                sin_r = np.sin(rot_rad)

                rotated_points = []
                for px, py in flake['points']:
                    # Rotate
                    rx = px * cos_r - py * sin_r
                    ry = px * sin_r + py * cos_r
                    # Translate to position
                    rotated_points.append((int(x + rx), int(y + ry)))

                # Draw filled polygon for irregular snowflake
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], (255, 255, 255), cv2.LINE_AA)

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

    # Mode toggle - start with effect ON
    effect_enabled = True

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
            # Effect mode: apply the effect
            matrix.update()
            result = matrix.draw(frame)
        else:
            # Preview mode: show raw webcam
            result = frame.copy()

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        # Display FPS only (no mode text)
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

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
    """Heat wave effect with thermal refraction"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Heat waves - horizontal cylinders with refraction
        self.heat_waves = []
        for _ in range(8):  # 8 heat waves at various positions
            # Create slightly irregular horizontal cylinder
            wave_width = random.randint(width // 3, width // 2)
            wave_height = random.randint(40, 80)

            # Precompute refraction displacement map for this heat wave
            # Create a lens-like distortion pattern
            displacement_map = np.zeros((wave_height, wave_width, 2), dtype=np.float32)

            for y in range(wave_height):
                for x in range(wave_width):
                    # Distance from center of wave (normalized)
                    center_y = wave_height / 2.0
                    dist_from_center = abs(y - center_y) / center_y

                    # Create refraction effect - strongest at center, weak at edges
                    # Use sine wave for smooth falloff
                    strength = (1.0 - dist_from_center) * np.sin(dist_from_center * np.pi)

                    # Horizontal displacement (wave-like pattern)
                    x_offset = strength * np.sin(x * 0.1 + y * 0.05) * 8.0

                    # Vertical displacement (slight upward distortion)
                    y_offset = strength * 3.0

                    displacement_map[y, x] = [x_offset, y_offset]

            # Blur the displacement map for smoother refraction
            displacement_map[:, :, 0] = cv2.GaussianBlur(displacement_map[:, :, 0], (15, 15), 0)
            displacement_map[:, :, 1] = cv2.GaussianBlur(displacement_map[:, :, 1], (15, 15), 0)

            self.heat_waves.append({
                'x': random.randint(0, width - wave_width),
                'y': random.randint(height, height + 200),  # Start below screen
                'speed': random.uniform(0.8, 2.0),  # Upward speed
                'width': wave_width,
                'height': wave_height,
                'displacement_map': displacement_map,
                'opacity': random.uniform(0.6, 0.9)
            })

    def update(self):
        """Update heat waves"""
        for wave in self.heat_waves:
            # Move upward
            wave['y'] -= wave['speed']

            # Reset if off screen (disappeared near middle or above)
            if wave['y'] < -wave['height']:
                # Restart from bottom
                wave['y'] = random.randint(self.height, self.height + 200)
                wave['x'] = random.randint(0, self.width - wave['width'])
                wave['speed'] = random.uniform(0.8, 2.0)

    def draw(self, frame, face_mask=None):
        """Golden sunset effect with thermal heat waves - replace hue with golden sunlight while preserving S and V"""
        # Apply heat wave refraction first
        result = frame.copy()

        for wave in self.heat_waves:
            wave_x = int(wave['x'])
            wave_y = int(wave['y'])
            wave_w = wave['width']
            wave_h = wave['height']

            # Only process if wave is visible on screen
            if wave_y < self.height and wave_y + wave_h > 0:
                # Calculate visible portion of wave
                src_y_start = max(0, -wave_y)
                src_y_end = min(wave_h, self.height - wave_y)
                dst_y_start = max(0, wave_y)
                dst_y_end = min(self.height, wave_y + wave_h)

                src_x_start = max(0, -wave_x)
                src_x_end = min(wave_w, self.width - wave_x)
                dst_x_start = max(0, wave_x)
                dst_x_end = min(self.width, wave_x + wave_w)

                if src_y_end > src_y_start and src_x_end > src_x_start:
                    # Get the region to refract
                    region = result[dst_y_start:dst_y_end, dst_x_start:dst_x_end].copy()
                    region_h, region_w = region.shape[:2]

                    # Get corresponding displacement map section
                    disp_map = wave['displacement_map'][src_y_start:src_y_end, src_x_start:src_x_end]

                    # Apply refraction using displacement map
                    for y in range(region_h):
                        for x in range(region_w):
                            if y < disp_map.shape[0] and x < disp_map.shape[1]:
                                dx, dy = disp_map[y, x]

                                # Calculate source pixel with displacement
                                src_x = int(x + dx)
                                src_y = int(y + dy)

                                # Clamp to region bounds
                                src_x = max(0, min(region_w - 1, src_x))
                                src_y = max(0, min(region_h - 1, src_y))

                                # Copy refracted pixel (blend with opacity)
                                if 0 <= src_y < region_h and 0 <= src_x < region_w:
                                    result[dst_y_start + y, dst_x_start + x] = region[src_y, src_x]

        # Convert to HSV to manipulate hue
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Golden sunset hue in OpenCV (0-179 range)
        # Golden/orange sunset is around 15-25 in HSV
        # Using 20 for warm golden sunlight
        golden_hue = 20

        # Replace all hues with golden sunset hue, keep original S and V
        hsv[:, :, 0] = golden_hue

        # Convert back to BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Apply gradient of golden sunlight - gentle gradient down to halfway
        height, width = result.shape[:2]

        # Create gradient mask - gentler, extends to halfway down
        gradient = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            # Smooth gradient from 0.5 at top to 0.0 at halfway down
            if y < height // 2:
                # Linear fade from 0.5 to 0.0
                gradient[y, :] = 0.5 * (1.0 - (y / (height // 2)))
            else:
                gradient[y, :] = 0.0

        # Convert gradient to 3-channel
        gradient_3channel = cv2.merge([gradient, gradient, gradient])

        # Create bright golden sunlight color
        bright_golden = np.ones_like(result, dtype=np.uint8)
        bright_golden[:, :] = [100, 200, 255]  # Very bright golden/yellow

        # Blend sunlight gradient on top of result
        result = (result.astype(np.float32) * (1.0 - gradient_3channel) +
                  bright_golden.astype(np.float32) * gradient_3channel).astype(np.uint8)

        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Blur edges for soft effect
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges, edges, edges])

        # Create golden color (BGR format)
        # Golden/orange sunset color
        golden = np.ones_like(result, dtype=np.uint8)
        golden[:, :] = [0, 165, 255]  # BGR: bright golden orange

        # Blend golden edges on top of golden frame
        alpha = edges_3channel.astype(np.float32) / 255.0
        result = (result.astype(np.float32) * (1.0 - alpha) + golden.astype(np.float32) * alpha).astype(np.uint8)

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
            print(f"  {i+1}. Camera {cam['id']} - {cam['width']}x{cam['height']}")

        while True:
            try:
                choice = input(f"\nSelect camera (1-{len(available_cameras)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_cameras):
                    selected_camera = available_cameras[choice_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_cameras)}")
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

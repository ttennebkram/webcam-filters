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
    """Melting effect - water drops refract the image as they drip down"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Pre-calculate refraction maps for different drop sizes
        self.drop_sizes = [8, 10, 12, 15, 18]  # Smaller drop widths
        self.refraction_maps = {}

        for size in self.drop_sizes:
            self.refraction_maps[size] = self.create_refraction_map(size)

        # Active drops - each drop has position, size, and speed
        self.drops = []

        # Spawn initial drops
        for _ in range(1000):
            self.spawn_drop()

    def create_refraction_map(self, drop_width):
        """Pre-calculate refraction displacement map for a water drop shape"""
        drop_height = int(drop_width * 2.0)  # Drops are elongated
        half_w = drop_width // 2

        # Create displacement maps and alpha mask
        offset_x = np.zeros((drop_height, drop_width), dtype=np.float32)
        offset_y = np.zeros((drop_height, drop_width), dtype=np.float32)
        alpha = np.zeros((drop_height, drop_width), dtype=np.float32)

        # Create water drop shape with refraction effect
        for y in range(drop_height):
            for x in range(drop_width):
                # Distance from center x-axis
                dx = x - half_w
                fx = x - half_w + 0.5  # Sub-pixel center

                # Normalize y position (0 at top, 1 at bottom)
                ny = y / drop_height

                # Drop shape: stretched tail at top, rounded bulb at bottom
                if ny < 0.5:
                    # Top half - narrow stretched tail
                    max_radius = half_w * (0.2 + 0.5 * (ny / 0.5))
                    # Tail has lower alpha - lets things through
                    alpha_strength = 0.3 + 0.5 * (ny / 0.5)
                elif ny < 0.85:
                    # Middle - widening to rounded bottom
                    progress = (ny - 0.5) / 0.35
                    # Use sine curve for smooth rounding
                    max_radius = half_w * (0.7 + 0.3 * np.sin(progress * np.pi / 2))
                    alpha_strength = 0.8 + 0.2 * progress
                else:
                    # Bottom tip - smooth rounded point using cosine
                    progress = (ny - 0.85) / 0.15
                    # Smooth curve to a point
                    max_radius = half_w * np.cos(progress * np.pi / 2)
                    alpha_strength = 1.0

                # Use floating point distance for smoother edges
                dist_from_center = np.sqrt(fx * fx)

                if dist_from_center < max_radius:
                    # Inside the drop - apply refraction
                    # Refraction strength based on distance from edge
                    edge_dist = max_radius - dist_from_center
                    strength = edge_dist / max_radius

                    # Strong refraction at edges (like real water drops)
                    # Center has minimal distortion, edges have maximum
                    if strength < 0.3:
                        # Near edge - very strong refraction with smooth gradient
                        refract_strength = 60.0 * (1.0 - strength / 0.3)
                    else:
                        # Center - moderate refraction
                        refract_strength = 15.0

                    # Smooth falloff at edges for anti-aliasing - wider feathering
                    edge_falloff = min(1.0, edge_dist / 4.0)  # Wider feather (was /2.0)

                    offset_x[y, x] = np.sign(dx) * refract_strength
                    offset_y[y, x] = refract_strength * 0.3
                    alpha[y, x] = edge_falloff * alpha_strength

        return {
            'width': drop_width,
            'height': drop_height,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'alpha': alpha
        }

    def spawn_drop(self):
        """Create a new drop at random position"""
        drop = {
            'x': random.randint(0, self.width - 1),
            'y': random.randint(-100, 0),  # Start above screen
            'size': random.choice(self.drop_sizes),
            'speed': random.uniform(8.0, 24.0)  # Double average speed
        }
        self.drops.append(drop)

    def update(self):
        """Update drop positions"""
        # Move drops down
        for drop in self.drops:
            drop['y'] += drop['speed']

        # Remove drops that are off screen and spawn new ones
        self.drops = [d for d in self.drops if d['y'] < self.height + 50]

        # Maintain drop count
        while len(self.drops) < 1000:
            self.spawn_drop()

    def draw(self, frame, face_mask=None):
        """Draw water drops refracting the background image"""
        # Boost saturation and contrast of source frame for more visible drops
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.3, beta=0)  # 1.3x saturation
        hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=1.15, beta=-10)  # Subtle contrast boost
        enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Start with original frame (not enhanced)
        result = frame.copy()

        # Sort drops by y position (top to bottom) so overlapping drops blend correctly
        sorted_drops = sorted(self.drops, key=lambda d: d['y'])

        # Apply each drop's refraction with alpha blending
        for drop in sorted_drops:
            drop_x = int(drop['x'])
            drop_y = int(drop['y'])
            size = drop['size']

            # Get pre-calculated refraction map
            refraction = self.refraction_maps[size]
            drop_width = refraction['width']
            drop_height = refraction['height']
            offset_x = refraction['offset_x']
            offset_y = refraction['offset_y']
            alpha = refraction['alpha']

            # Calculate drop bounds on screen
            left = drop_x - drop_width // 2
            right = left + drop_width
            top = drop_y
            bottom = drop_y + drop_height

            # Skip if completely off screen
            if right < 0 or left >= self.width or bottom < 0 or top >= self.height:
                continue

            # Clip to screen bounds
            screen_left = max(0, left)
            screen_right = min(self.width, right)
            screen_top = max(0, top)
            screen_bottom = min(self.height, bottom)

            # Map to drop coordinates
            drop_left_offset = screen_left - left
            drop_right_offset = drop_left_offset + (screen_right - screen_left)
            drop_top_offset = screen_top - top
            drop_bottom_offset = drop_top_offset + (screen_bottom - screen_top)

            # Extract the region from refraction maps
            region_offset_x = offset_x[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]
            region_offset_y = offset_y[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]
            region_alpha = alpha[drop_top_offset:drop_bottom_offset, drop_left_offset:drop_right_offset]

            # Create coordinate grids for this region
            region_h = screen_bottom - screen_top
            region_w = screen_right - screen_left

            y_coords, x_coords = np.mgrid[screen_top:screen_bottom, screen_left:screen_right]

            # Calculate source coordinates with refraction
            source_x = (x_coords + region_offset_x.astype(np.int32)).clip(0, self.width - 1)
            source_y = (y_coords + region_offset_y.astype(np.int32)).clip(0, self.height - 1)

            # Get refracted pixels from ENHANCED frame (saturated/contrasted)
            refracted = enhanced_frame[source_y, source_x]

            # Get original pixels
            original = result[screen_top:screen_bottom, screen_left:screen_right]

            # Alpha blend (vectorized)
            alpha_3d = region_alpha[:, :, np.newaxis]  # Add channel dimension
            blended = (refracted * alpha_3d + original * (1.0 - alpha_3d)).astype(np.uint8)

            # Only update pixels with significant alpha
            mask = region_alpha > 0.01
            result[screen_top:screen_bottom, screen_left:screen_right][mask] = blended[mask]

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
    print("  SPACEBAR - Toggle Matrix mode on/off")
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
    matrix_mode = False  # Start in preview mode

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

        if matrix_mode:
            # Matrix mode: update grid and draw based on brightness
            matrix.update()
            result = matrix.draw(frame)
            mode_text = "MATRIX MODE"
        else:
            # Preview mode: show raw webcam
            result = frame.copy()
            mode_text = "PREVIEW MODE - Press SPACEBAR for Matrix"

        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0

        # Display mode and FPS
        cv2.putText(result, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 60),
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
                matrix_mode = not matrix_mode
                if matrix_mode:
                    print("Matrix mode activated!")
                else:
                    print("Preview mode - showing raw webcam")
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

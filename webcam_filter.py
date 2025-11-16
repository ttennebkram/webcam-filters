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
    """Cut glass effect - static geometric facets that refract the image"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Create static cut-glass facet pattern
        # Facets are diamond/triangular shapes that create prismatic refraction
        self.facet_size = 120  # Size of each facet

        # Pre-calculate the entire cut-glass mask for the screen
        self.refraction_map = self.create_cut_glass_pattern()

    def create_cut_glass_pattern(self):
        """Pre-calculate refraction displacement map for entire screen with square-framed facets"""
        # Create displacement maps for entire screen
        offset_x = np.zeros((self.height, self.width), dtype=np.float32)
        offset_y = np.zeros((self.height, self.width), dtype=np.float32)
        alpha = np.zeros((self.height, self.width), dtype=np.float32)

        facet_size = self.facet_size

        # Create a regular grid of square facets
        for row in range(0, self.height, facet_size):
            for col in range(0, self.width, facet_size):
                # Create square facet at this position
                self.add_square_facet(offset_x, offset_y, alpha, col, row, facet_size)

        # Blur the displacement maps for smoother transitions between facets
        offset_x = cv2.GaussianBlur(offset_x, (5, 5), 0)
        offset_y = cv2.GaussianBlur(offset_y, (5, 5), 0)

        return {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'alpha': alpha
        }

    def add_square_facet(self, offset_x, offset_y, alpha, corner_x, corner_y, size):
        """Add a single square-framed facet with beveled edges and indented center"""
        # Calculate center of the square
        center_x = corner_x + size // 2
        center_y = corner_y + size // 2

        # Frame thickness
        frame_thickness = 0  # No dark frame between squares
        # Flat raised border width (no refraction) - 8 pixels
        flat_border = 8
        # Transition zone from flat to full refraction
        transition_zone = size // 16  # Smaller bevel slope (~7.5 pixels)

        for y in range(max(0, corner_y), min(self.height, corner_y + size)):
            for x in range(max(0, corner_x), min(self.width, corner_x + size)):
                # Distance from square boundaries
                dist_left = x - corner_x
                dist_right = (corner_x + size) - x
                dist_top = y - corner_y
                dist_bottom = (corner_y + size) - y

                # Distance from center
                dx = x - center_x
                dy = y - center_y

                # Minimum distance to any edge
                min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_edge_dist < frame_thickness:
                    # In the dark frame - no refraction
                    alpha[y, x] = 0.0
                elif min_edge_dist < flat_border:
                    # In the flat raised border area - minimal or no refraction
                    offset_x[y, x] = 0.0
                    offset_y[y, x] = 0.0
                    alpha[y, x] = 0.0  # Flat, no distortion
                elif min_edge_dist < flat_border + transition_zone:
                    # In bevel zone - refraction goes from center toward the raised edge
                    # This is the slope from valley (center) up to raised flat border
                    transition_progress = (min_edge_dist - flat_border) / transition_zone

                    # Inverse progress: 1.0 at inner edge (near valley), 0.0 at outer edge (near flat)
                    bevel_strength = 1.0 - transition_progress

                    # Refraction pushes TOWARD the nearest edge (upward slope effect)
                    # Determine which edge is closest
                    if dist_left == min_edge_dist:
                        # Left edge is closest - push toward left
                        refract_x = -35.0 * bevel_strength
                        refract_y = 0.0
                    elif dist_right == min_edge_dist:
                        # Right edge is closest - push toward right
                        refract_x = 35.0 * bevel_strength
                        refract_y = 0.0
                    elif dist_top == min_edge_dist:
                        # Top edge is closest - push toward top
                        refract_x = 0.0
                        refract_y = -35.0 * bevel_strength
                    else:
                        # Bottom edge is closest - push toward bottom
                        refract_x = 0.0
                        refract_y = 35.0 * bevel_strength

                    offset_x[y, x] = refract_x
                    offset_y[y, x] = refract_y
                    alpha[y, x] = 1.0 * bevel_strength
                else:
                    # In the indented valley center - create crosshatch diagonal pattern
                    # Diagonal lines run at 45 degrees in both directions
                    diagonal_spacing = 40  # Space between diagonal ridges (reduced number by half again)
                    diagonal_width = 1  # Width of each diagonal ridge (slightly rounded top)

                    # Calculate position along both diagonals (45 degree angles)
                    diagonal_pos1 = (dx + dy) % diagonal_spacing  # One direction
                    diagonal_pos2 = (dx - dy) % diagonal_spacing  # Other direction

                    # Distance from center of diagonal ridges in both directions
                    dist_from_ridge1 = abs(diagonal_pos1 - diagonal_spacing / 2)
                    dist_from_ridge2 = abs(diagonal_pos2 - diagonal_spacing / 2)

                    # Use the minimum distance (closer to either diagonal)
                    dist_from_ridge = min(dist_from_ridge1, dist_from_ridge2)

                    if dist_from_ridge < diagonal_width:
                        # On a diagonal ridge - more pronounced elevation
                        # Gentle rounding (1-4 pixels of curvature)
                        ridge_roundness = 1.0 - (dist_from_ridge / diagonal_width)
                        ridge_height = ridge_roundness * 0.6  # More pronounced elevation

                        # Very minimal refraction on ridge tops for sharper effect
                        refract_x = 3.0 * (1.0 - ridge_height)
                        refract_y = 3.0 * (1.0 - ridge_height)
                        alpha[y, x] = 0.2 + 0.5 * ridge_height
                    else:
                        # Between ridges - valley bottom with strong refraction for visibility
                        # Distance to nearest ridge
                        valley_depth = (dist_from_ridge - diagonal_width) / (diagonal_spacing / 2 - diagonal_width)
                        valley_depth = min(1.0, valley_depth)

                        # Much stronger refraction in valleys for dramatic visibility
                        if abs(dx) > abs(dy):
                            if dx > 0:
                                refract_x = 70.0 * valley_depth
                                refract_y = (dy / abs(dx) * 45.0 * valley_depth) if abs(dx) > 0 else 0.0
                            else:
                                refract_x = -70.0 * valley_depth
                                refract_y = (dy / abs(dx) * 45.0 * valley_depth) if abs(dx) > 0 else 0.0
                        else:
                            if dy > 0:
                                refract_y = 70.0 * valley_depth
                                refract_x = (dx / abs(dy) * 45.0 * valley_depth) if abs(dy) > 0 else 0.0
                            else:
                                refract_y = -70.0 * valley_depth
                                refract_x = (dx / abs(dy) * 45.0 * valley_depth) if abs(dy) > 0 else 0.0

                        alpha[y, x] = 1.0 * valley_depth

                    offset_x[y, x] = refract_x
                    offset_y[y, x] = refract_y

    def update(self):
        """Update animation - static effect, no updates needed"""
        pass

    def draw(self, frame, face_mask=None):
        """Apply cut-glass facet refraction to the entire frame"""
        # Use original frame without enhancement
        enhanced_frame = frame.copy()

        # Get pre-calculated refraction maps
        offset_x = self.refraction_map['offset_x']
        offset_y = self.refraction_map['offset_y']
        alpha_map = self.refraction_map['alpha']

        # Create coordinate grids for entire frame
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]

        # Calculate source coordinates with refraction
        source_x = (x_coords + offset_x.astype(np.int32)).clip(0, self.width - 1)
        source_y = (y_coords + offset_y.astype(np.int32)).clip(0, self.height - 1)

        # Get refracted pixels from enhanced frame
        refracted = enhanced_frame[source_y, source_x]

        # Alpha blend with original frame
        alpha_3d = alpha_map[:, :, np.newaxis]  # Add channel dimension
        result = (refracted * alpha_3d + frame * (1.0 - alpha_3d)).astype(np.uint8)

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

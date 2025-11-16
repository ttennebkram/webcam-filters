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
    """Spring effect"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Generate grass blades along bottom
        self.grass_blades = []
        grass_height_max = 50  # Maximum grass height (shorter)
        num_blades = 200  # Dense grass

        for _ in range(num_blades):
            x = random.randint(0, width)
            # Grass height varies
            height_offset = random.randint(25, grass_height_max)
            # More varied angles for natural look
            angle = random.uniform(-45, 45)
            # Width variation
            blade_width = random.randint(2, 4)
            # Color variation - darker hunter green base
            color_variance = random.randint(-15, 15)
            base_color = np.array([20, 60, 20])  # Darker hunter green
            color = np.clip(base_color + color_variance, 0, 255)

            self.grass_blades.append({
                'x': x,
                'height': height_offset,
                'angle': angle,
                'width': blade_width,
                'color': tuple(map(int, color))
            })

        # Generate Easter eggs
        self.easter_eggs = []
        num_eggs = 8

        for _ in range(num_eggs):
            # Position near bottom, in grass area
            x = random.randint(50, width - 50)
            y = height - random.randint(20, 50)  # Nestled in grass

            # Size variation - 50% larger
            egg_width = random.randint(30, 45)
            egg_height = int(egg_width * 1.4)  # Eggs are taller than wide

            # Random orientation angle
            rotation_angle = random.randint(0, 360)

            # Egg colors - pastel spring colors
            color_choices = [
                [180, 150, 255],  # Pastel pink (BGR)
                [255, 200, 150],  # Pastel blue
                [150, 220, 255],  # Pastel yellow
                [180, 255, 200],  # Pastel green
                [220, 180, 220],  # Pastel purple
                [200, 220, 255],  # Pale peach
            ]
            base_color = random.choice(color_choices)

            # Pattern type
            pattern = random.choice(['dots', 'stripes', 'solid'])

            self.easter_eggs.append({
                'x': x,
                'y': y,
                'width': egg_width,
                'height': egg_height,
                'color': base_color,
                'pattern': pattern,
                'rotation': rotation_angle
            })

    def update(self):
        """Update animation"""
        pass

    def draw(self, frame, face_mask=None):
        """Spring effect with light sky blue gradient and hunter green edges"""
        result = frame.copy()

        # Apply gradient of light sky blue - gentle gradient down to halfway
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

        # Create light sky blue color
        light_sky_blue = np.ones_like(result, dtype=np.uint8)
        light_sky_blue[:, :] = [235, 206, 135]  # BGR: light sky blue

        # Blend sky blue gradient on top of result
        result = (result.astype(np.float32) * (1.0 - gradient_3channel) +
                  light_sky_blue.astype(np.float32) * gradient_3channel).astype(np.uint8)

        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to make them bolder
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Blur edges for soft effect
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Boost edge intensity to make them more prominent
        edges = cv2.convertScaleAbs(edges, alpha=1.5, beta=0)
        edges = np.clip(edges, 0, 255).astype(np.uint8)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges, edges, edges])

        # Create hunter green color (BGR format)
        hunter_green = np.ones_like(result, dtype=np.uint8)
        hunter_green[:, :] = [35, 86, 35]  # BGR: hunter green

        # Blend hunter green edges on top of frame with stronger alpha
        alpha = (edges_3channel.astype(np.float32) / 255.0) * 1.3
        alpha = np.clip(alpha, 0, 1)
        result = (result.astype(np.float32) * (1.0 - alpha) + hunter_green.astype(np.float32) * alpha).astype(np.uint8)

        # Draw Easter eggs first (so grass can overlap them)
        for egg in self.easter_eggs:
            cx, cy = egg['x'], egg['y']
            w, h = egg['width'], egg['height']
            rotation = egg['rotation']

            # Create egg shape (ellipse) with rotation
            overlay = result.copy()

            # Main egg color with subtle shading (rotated ellipse)
            cv2.ellipse(overlay, (cx, cy), (w // 2, h // 2), rotation, 0, 360, egg['color'], -1)

            # Add subtle highlight for dimension (rotated offset)
            highlight_angle = np.radians(rotation - 45)
            highlight_offset_x = int(w // 6 * np.cos(highlight_angle))
            highlight_offset_y = int(h // 6 * np.sin(highlight_angle))
            highlight_color = [min(255, c + 40) for c in egg['color']]
            cv2.ellipse(overlay, (cx - highlight_offset_x, cy - highlight_offset_y),
                       (w // 4, h // 4), rotation, 0, 360, highlight_color, -1)

            # Add subtle shadow (bottom-right, accounting for rotation)
            shadow_angle = np.radians(rotation + 135)
            shadow_offset_x = int(w // 8 * np.cos(shadow_angle))
            shadow_offset_y = int(h // 8 * np.sin(shadow_angle))
            shadow_color = [max(0, c - 30) for c in egg['color']]
            cv2.ellipse(overlay, (cx - shadow_offset_x, cy - shadow_offset_y),
                       (w // 3, h // 3), rotation, 0, 360, shadow_color, -1)

            # Add pattern
            if egg['pattern'] == 'dots':
                # Small dots
                for _ in range(random.randint(6, 10)):
                    dot_x = cx + random.randint(-w // 3, w // 3)
                    dot_y = cy + random.randint(-h // 3, h // 3)
                    dot_color = [max(0, c - 40) for c in egg['color']]
                    cv2.circle(overlay, (dot_x, dot_y), 3, dot_color, -1)

            elif egg['pattern'] == 'stripes':
                # Stripes perpendicular to egg orientation
                stripe_color = [max(0, c - 40) for c in egg['color']]
                for i in range(3):
                    stripe_offset = -h // 3 + i * h // 3
                    # Calculate stripe position along egg's major axis
                    stripe_x = int(cx + stripe_offset * np.sin(np.radians(rotation)))
                    stripe_y = int(cy - stripe_offset * np.cos(np.radians(rotation)))
                    cv2.ellipse(overlay, (stripe_x, stripe_y), (w // 2, 4),
                               rotation + 90, 0, 360, stripe_color, -1)

            # Blend egg with soft edges
            alpha_egg = 0.85
            result = cv2.addWeighted(overlay, alpha_egg, result, 1 - alpha_egg, 0)

        # Draw grass blades
        for blade in self.grass_blades:
            x = blade['x']
            base_y = self.height
            tip_y = self.height - blade['height']

            # Calculate slight curve and angle
            angle_rad = np.radians(blade['angle'])
            tip_x = int(x + blade['height'] * np.sin(angle_rad) * 0.3)

            # Draw blade as a thin tapered line
            # Create points for a tapered blade shape
            thickness = blade['width']

            # Draw main blade
            cv2.line(result, (x, base_y), (tip_x, tip_y), blade['color'], thickness)

            # Add slight highlight on one side for depth
            highlight_color = tuple(min(255, c + 20) for c in blade['color'])
            cv2.line(result, (x, base_y), (tip_x, tip_y), highlight_color, max(1, thickness - 1))

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

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

        # Christmas light colors - fully saturated
        self.christmas_colors = [
            (0, 0, 255),      # Pure red
            (0, 255, 0),      # Pure green
            (255, 0, 0),      # Pure blue
            (0, 215, 255),    # Pure gold/yellow
            (255, 0, 255),    # Pure magenta
            (0, 255, 255),    # Pure yellow
            (255, 100, 0),    # Pure cyan/light blue
            (128, 0, 255),    # Pure orange-red
        ]

        # Ornament ball colors - reflective appearance
        ornament_colors = [
            (0, 0, 200),      # Deep red
            (0, 180, 0),      # Deep green
            (0, 180, 220),    # Deep gold
        ]

        # Pre-generate ornament balls with fixed positions and colors
        self.ornament_balls = []
        garland_depth = 50

        # Top border balls
        for i in range(0, width, 80):
            if i > 40 and i < width - 40:
                wave_offset = int(10 * np.sin(i * 0.05))
                self.ornament_balls.append({
                    'x': i,
                    'y': garland_depth + wave_offset - 5,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Bottom border balls (offset)
        for i in range(40, width, 80):
            if i > 40 and i < width - 40:
                wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
                self.ornament_balls.append({
                    'x': i,
                    'y': height - garland_depth + wave_offset + 5,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Left border balls
        for i in range(0, height, 80):
            if i > 40 and i < height - 40:
                wave_offset = int(10 * np.sin(i * 0.05))
                self.ornament_balls.append({
                    'x': garland_depth + wave_offset - 5,
                    'y': i,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Right border balls (offset)
        for i in range(40, height, 80):
            if i > 40 and i < height - 40:
                wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
                self.ornament_balls.append({
                    'x': width - garland_depth + wave_offset + 5,
                    'y': i,
                    'size': random.randint(12, 18),
                    'color': random.choice(ornament_colors)
                })

        # Snowflakes - irregular shapes with christmas colors
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
                'rotation': random.uniform(0, 360),
                'color': random.choice(self.christmas_colors)  # Random christmas color
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

                # Regenerate irregular shape and new color
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
                flake['color'] = random.choice(self.christmas_colors)

    def draw(self, frame, face_mask=None):
        """Christmas effect - warm glow with saturated gold edges"""
        # Apply warm tungsten bulb color temperature
        # Convert to LAB color space for better color temperature control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Shift colors toward warm tungsten (yellow-orange)
        # A channel: green-red, B channel: blue-yellow
        lab[:, :, 1] = lab[:, :, 1] + 10  # Shift slightly toward red
        lab[:, :, 2] = lab[:, :, 2] + 35  # Shift strongly toward yellow/orange

        # Clamp values
        lab = np.clip(lab, 0, 255)

        # Convert back to BGR
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Detect edges - lower thresholds to keep more edges stable
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)  # Lower thresholds for more stable edges

        # Dilate edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Blur edges more at the edges for softer falloff
        edges_blurred = cv2.GaussianBlur(edges_dilated, (15, 15), 0)

        # Convert edges to 3-channel for blending
        edges_3channel = cv2.merge([edges_blurred, edges_blurred, edges_blurred])

        # Create very bright saturated GOLD for edges
        saturated_gold = np.ones_like(result, dtype=np.uint8)
        saturated_gold[:, :] = [0, 215, 255]  # Saturated gold (BGR)

        # Blend saturated gold on edges with stronger alpha for more visible gold
        alpha = edges_3channel.astype(np.float32) / 255.0 * 1.5  # Boost alpha for more gold
        alpha = np.clip(alpha, 0, 1.0)  # Clamp to valid range
        result = (result.astype(np.float32) * (1.0 - alpha) +
                 saturated_gold.astype(np.float32) * alpha).astype(np.uint8)

        # Draw pine garland border around frame - solid bushy appearance
        garland_depth = 50
        hunter_green_dark = (15, 70, 35)  # Dark hunter green
        hunter_green = (20, 85, 45)  # Medium hunter green
        hunter_green_light = (25, 95, 55)  # Light hunter green

        # Create dense pine garland on all four sides
        for i in range(0, self.width, 3):
            # Top border
            wave_offset = int(10 * np.sin(i * 0.05))
            depth = garland_depth + wave_offset

            # Draw overlapping foliage for dense appearance
            for offset in range(-2, 3):
                y_pos = depth + offset * 3
                # Random needle clusters
                for _ in range(3):
                    needle_x = i + random.randint(-4, 4)
                    needle_y = y_pos + random.randint(-6, 6)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle - np.pi/2))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle - np.pi/2))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

            # Bottom border
            wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
            depth = self.height - garland_depth + wave_offset

            for offset in range(-2, 3):
                y_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = i + random.randint(-4, 4)
                    needle_y = y_pos + random.randint(-6, 6)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle + np.pi/2))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle + np.pi/2))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

        for i in range(0, self.height, 3):
            # Left border
            wave_offset = int(10 * np.sin(i * 0.05))
            depth = garland_depth + wave_offset

            for offset in range(-2, 3):
                x_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = x_pos + random.randint(-6, 6)
                    needle_y = i + random.randint(-4, 4)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle - np.pi))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle - np.pi))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

            # Right border
            wave_offset = int(10 * np.sin(i * 0.05 + np.pi))
            depth = self.width - garland_depth + wave_offset

            for offset in range(-2, 3):
                x_pos = depth + offset * 3
                for _ in range(3):
                    needle_x = x_pos + random.randint(-6, 6)
                    needle_y = i + random.randint(-4, 4)
                    needle_len = random.randint(8, 15)
                    needle_angle = random.uniform(-np.pi/3, np.pi/3)

                    end_x = int(needle_x + needle_len * np.cos(needle_angle))
                    end_y = int(needle_y + needle_len * np.sin(needle_angle))

                    color = random.choice([hunter_green_dark, hunter_green, hunter_green_light])
                    cv2.line(result, (needle_x, needle_y), (end_x, end_y), color, 2, cv2.LINE_AA)

        # Draw ornament balls with reflective appearance
        for ball in self.ornament_balls:
            center = (ball['x'], ball['y'])
            radius = ball['size']
            base_color = ball['color']

            # Draw main ball with gradient effect for reflection
            # Create multiple circles with decreasing intensity for smooth gradient
            for i in range(radius, 0, -1):
                # Calculate color interpolation from bright center to darker edges
                t = i / radius
                # Darker base color
                color = tuple(int(c * (0.6 + 0.4 * t)) for c in base_color)
                cv2.circle(result, center, i, color, -1, cv2.LINE_AA)

            # Add bright highlight spot (upper left) for glossy reflection
            highlight_offset_x = int(-radius * 0.35)
            highlight_offset_y = int(-radius * 0.35)
            highlight_center = (ball['x'] + highlight_offset_x, ball['y'] + highlight_offset_y)
            highlight_radius = max(2, int(radius * 0.4))

            # Bright version of the color for highlight
            highlight_color = tuple(min(255, int(c * 1.8)) for c in base_color)
            cv2.circle(result, highlight_center, highlight_radius, highlight_color, -1, cv2.LINE_AA)

            # Add small white specular highlight for extra shine
            specular_center = (ball['x'] + int(-radius * 0.25), ball['y'] + int(-radius * 0.25))
            specular_radius = max(1, int(radius * 0.2))
            cv2.circle(result, specular_center, specular_radius, (255, 255, 255), -1, cv2.LINE_AA)

            # Add subtle darker outline for depth
            outline_color = tuple(int(c * 0.4) for c in base_color)
            cv2.circle(result, center, radius, outline_color, 1, cv2.LINE_AA)

        # Draw snowflakes - irregular colored christmas lights
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

                # Draw filled polygon with christmas light color
                pts = np.array(rotated_points, dtype=np.int32)
                cv2.fillPoly(result, [pts], flake['color'], cv2.LINE_AA)

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

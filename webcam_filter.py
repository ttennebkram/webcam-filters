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
    """Matrix-style falling characters with face-aware flow physics"""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Matrix character set: ASCII only for maximum speed
        self.chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.¦|<>*+-=')

        # Column width for character rain - MAXIMUM streamers with OpenCV speed
        self.col_width = 4  # Ultra narrow = maximum streamers
        self.num_cols = width // self.col_width

        # Initialize rain columns
        self.drops = []
        self.char_spacing = 16  # Smaller vertical spacing for smaller characters

        for i in range(self.num_cols):
            num_chars = 20
            self.drops.append({
                'x': i * self.col_width,
                'y': random.randint(-height, 0),
                'speed': random.uniform(3.29, 13.16),  # 30% faster: 2.53*1.3=3.29, 10.12*1.3=13.16
                'chars': [random.choice(self.chars) for _ in range(num_chars)],
                'char_x_offsets': [0.0] * num_chars,  # Each character's lateral offset (independent)
                'char_x_velocities': [0.0] * num_chars  # Each character's lateral velocity
            })

    def update_drops(self, obstacle_mask=None):
        """Update drop positions with physics - each character independent like water"""
        for drop in self.drops:
            # Update vertical position
            drop['y'] += drop['speed']

            # Reset drop if it falls off screen
            if drop['y'] > self.height:
                drop['y'] = random.randint(-100, -10)
                num_chars = len(drop['chars'])
                drop['chars'] = [random.choice(self.chars) for _ in range(num_chars)]
                drop['char_x_offsets'] = [0.0] * num_chars
                drop['char_x_velocities'] = [0.0] * num_chars
                continue

            # Skip physics if drop is above screen (optimization)
            if drop['y'] < -400:
                continue

            # Update EACH character independently based on what's at its location
            for i in range(len(drop['chars'])):
                char_y = int(drop['y'] - (i * self.char_spacing))

                # Skip characters that are off screen
                if char_y < 0 or char_y >= self.height:
                    continue

                char_x = int(drop['x'] + drop['char_x_offsets'][i])

                # Check if THIS character is hitting an obstacle
                if obstacle_mask is not None and 0 <= char_x < self.width:
                    # Look ahead for collision
                    collision = False
                    is_horizontal_edge = False

                    # Look down from this character's position
                    for dy in range(1, 25):
                        check_y = char_y + dy
                        if check_y >= self.height:
                            break

                        if obstacle_mask[check_y, char_x] > 0:
                            collision = True

                            # Check if horizontal edge - even wider detection
                            h_run = 0
                            for dx in range(-25, 25):  # Even wider range
                                cx = char_x + dx
                                if 0 <= cx < self.width and obstacle_mask[check_y, cx] > 0:
                                    h_run += 1
                            if h_run > 2:  # Very low threshold - catch all angles
                                is_horizontal_edge = True
                            break

                    if collision:
                        # Determine flow direction - search much further
                        left_space = 0
                        right_space = 0

                        for dx in range(1, 90):  # Search much further (was 70)
                            if char_x - dx >= 0 and obstacle_mask[char_y, char_x - dx] == 0:
                                left_space += 1
                            else:
                                break

                        for dx in range(1, 90):  # Search much further (was 70)
                            if char_x + dx < self.width and obstacle_mask[char_y, char_x + dx] == 0:
                                right_space += 1
                            else:
                                break

                        # Apply VERY strong forces for maximum deflection
                        force = 20.0 if is_horizontal_edge else 15.0  # Much stronger!

                        if left_space > right_space:
                            drop['char_x_velocities'][i] -= force
                        else:
                            drop['char_x_velocities'][i] += force
                    else:
                        # Not colliding - dampen lateral movement slower (let it flow more)
                        drop['char_x_velocities'][i] *= 0.85

                # Apply velocity and less damping for more dramatic flow
                drop['char_x_offsets'][i] += drop['char_x_velocities'][i]
                drop['char_x_velocities'][i] *= 0.92  # Less damping for longer flow

                # Wider clamp for more dramatic movement
                drop['char_x_offsets'][i] = max(-250, min(250, drop['char_x_offsets'][i]))

    def draw(self, frame, face_mask=None):
        """Draw the Matrix rain effect using OpenCV for speed"""
        # Start with a black background
        result = np.zeros_like(frame)

        # Use OpenCV font - much faster than PIL
        font = cv2.FONT_HERSHEY_SIMPLEX

        for drop in self.drops:
            base_x = drop['x']
            y_pos = int(drop['y'])

            # Early exit if entire drop is offscreen (major optimization)
            if y_pos < -400 or y_pos > self.height + 100:
                continue

            # Draw trail of characters - each has its own independent position
            # Only process visible characters for speed
            for i, char in enumerate(drop['chars']):
                char_y = y_pos - (i * self.char_spacing)

                # Skip characters that are offscreen (major speed optimization)
                if char_y < 0 or char_y >= self.height:
                    continue

                # Each character has its own lateral offset (independent movement like water particles)
                char_x_offset = drop['char_x_offsets'][i]
                x_pos = int(base_x + char_x_offset)

                if 0 <= x_pos < self.width:
                    # Only head (i==0) is bold
                    if i == 0:
                        thickness = 2
                    else:
                        thickness = 1

                    # Proper fade from white at head to very dark at tail
                    if i == 0:
                        color = (220, 255, 255)  # BGR: white at head
                    elif i == 1:
                        color = (100, 255, 200)  # BGR: bright green
                    elif i == 2:
                        color = (0, 220, 0)  # BGR: green
                    elif i == 3:
                        color = (0, 180, 0)  # BGR: medium green
                    elif i == 4:
                        color = (0, 140, 0)  # BGR: fading
                    elif i == 5:
                        color = (0, 100, 0)  # BGR: darker
                    elif i < 10:
                        intensity = 100 - ((i - 5) * 15)  # Fade from 100 to 25
                        color = (0, max(25, intensity), 0)
                    else:
                        intensity = 25 - ((i - 10) * 2)  # Fade from 25 to very dark
                        color = (0, max(5, intensity), 0)

                    cv2.putText(result, char, (x_pos, char_y), font, 0.4, color, thickness, cv2.LINE_AA)

        # Brighten background significantly - 50% brightness
        darkened_frame = cv2.convertScaleAbs(frame, alpha=0.50, beta=0)  # 50% brightness

        # Add green tint to background
        green_tint = np.zeros_like(frame)
        green_tint[:, :, 1] = 45  # More green channel for visibility
        background = cv2.addWeighted(darkened_frame, 0.75, green_tint, 0.25, 0)

        # Composite characters on top - characters should be bright
        result = cv2.addWeighted(background, 1.0, result, 1.0, 0)

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
            # Matrix mode: detect edges/face and apply effect
            obstacle_mask = edge_detector.detect(frame)
            matrix.update_drops(obstacle_mask)
            result = matrix.draw(frame, obstacle_mask)
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

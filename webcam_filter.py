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

        # Matrix character set: ASCII only for maximum speed
        self.chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.¦|<>*+-=')

        # Character grid settings
        self.char_width = 10  # Horizontal spacing
        self.char_height = 18  # Vertical spacing
        self.num_cols = width // self.char_width
        self.num_rows = height // self.char_height

        # Fixed grid of characters - each position has a character
        self.grid = []
        for row in range(self.num_rows):
            grid_row = []
            for col in range(self.num_cols):
                grid_row.append({
                    'char': random.choice(self.chars),
                    'change_counter': random.randint(0, 30)  # When to change character
                })
            self.grid.append(grid_row)

        # Streamers - waves of illumination moving down columns
        self.streamers = []
        for i in range(self.num_cols):
            if random.random() < 0.3:  # 30% of columns have active streamers
                self.streamers.append({
                    'col': i,
                    'row': random.randint(-20, 0),
                    'speed': random.uniform(0.3, 1.2),  # Rows per frame
                    'length': random.randint(10, 25)  # Length of trail
                })
            else:
                self.streamers.append(None)

    def update(self):
        """Update character grid and streamers"""
        # Randomly change characters in the grid
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell = self.grid[row][col]
                cell['change_counter'] -= 1
                if cell['change_counter'] <= 0:
                    cell['char'] = random.choice(self.chars)
                    cell['change_counter'] = random.randint(20, 40)

        # Update streamers
        for i, streamer in enumerate(self.streamers):
            if streamer is not None:
                # Move streamer down
                streamer['row'] += streamer['speed']

                # Reset if off screen
                if streamer['row'] > self.num_rows + streamer['length']:
                    streamer['row'] = random.randint(-30, -5)
                    streamer['speed'] = random.uniform(0.3, 1.2)
                    streamer['length'] = random.randint(10, 25)
            else:
                # Randomly spawn new streamers
                if random.random() < 0.002:  # Small chance each frame
                    self.streamers[i] = {
                        'col': i,
                        'row': random.randint(-20, 0),
                        'speed': random.uniform(0.3, 1.2),
                        'length': random.randint(10, 25)
                    }

    def draw(self, frame, face_mask=None):
        """Draw the Matrix effect - characters brightness based on background"""
        # Convert to HSV to manipulate saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Boost saturation significantly
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5, beta=0)  # 2.5x saturation

        # Convert back to BGR
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Dim the saturated frame for background
        background = cv2.convertScaleAbs(saturated_frame, alpha=0.3, beta=0)

        # Create edge-detected overlay with color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Soften edges with blur
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Create colored edge overlay by masking saturated frame with edges
        edge_mask = cv2.merge([edges, edges, edges])  # 3-channel mask
        edge_overlay = cv2.bitwise_and(saturated_frame, edge_mask)
        edge_overlay = cv2.convertScaleAbs(edge_overlay, alpha=0.45, beta=0)

        # Combine background with colored edge overlay
        background = cv2.addWeighted(background, 1.0, edge_overlay, 1.0, 0)

        # Start with this background
        result = background.copy()

        # Use OpenCV font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Create character layer
        char_layer = np.zeros_like(frame)

        # Draw all characters in the grid
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                x_pos = col * self.char_width
                y_pos = row * self.char_height + self.char_height  # Baseline

                if y_pos < 0 or y_pos >= self.height or x_pos >= self.width:
                    continue

                # Sample color and brightness from saturated background at this position
                brightness = gray[min(y_pos, self.height-1), min(x_pos, self.width-1)]
                bg_color = saturated_frame[min(y_pos, self.height-1), min(x_pos, self.width-1)]

                # Check if this position is part of a streamer
                streamer_intensity = 0
                for streamer in self.streamers:
                    if streamer is not None and streamer['col'] == col:
                        # Distance from streamer head (negative = behind, positive = ahead)
                        distance = streamer['row'] - row

                        if distance >= 0 and distance < streamer['length']:
                            if distance == 0:
                                # Head of streamer - white/bright
                                streamer_intensity = 255
                            else:
                                # Tail - fades from bright to dark BEHIND the head
                                fade = 1.0 - (distance / streamer['length'])
                                streamer_intensity = int(200 * fade)

                # Combine background brightness with streamer intensity
                if streamer_intensity > 0:
                    # Streamer - use white for head, colored for tail
                    if streamer_intensity > 240:
                        # Head - white
                        color = (255, 255, 255)
                        thickness = 2
                    else:
                        # Tail - take hue from background, boost saturation
                        intensity_scale = streamer_intensity / 200.0
                        color = tuple(int(c * intensity_scale * 1.5) for c in bg_color)
                        thickness = 1
                else:
                    # No streamer - characters use background color, saturated
                    intensity_scale = brightness / 255.0 * 1.2  # Boost brightness
                    color = tuple(int(c * intensity_scale) for c in bg_color)
                    thickness = 1

                # Check if color is too dark
                if max(color) < 20:
                    continue  # Don't draw very dark characters

                # Get character
                char = self.grid[row][col]['char']

                cv2.putText(char_layer, char, (x_pos, y_pos), font, 0.5, color, thickness, cv2.LINE_AA)

        # Composite characters on top of background
        result = cv2.addWeighted(result, 1.0, char_layer, 1.0, 0)

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

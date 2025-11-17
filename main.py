#!/usr/bin/env python3
"""
Unified webcam filter system.

Load and run any effect from the command line.
"""

import argparse
import sys
import tkinter as tk
import cv2

from effects import discover_effects, list_effects, get_effect_class
from core.camera import find_cameras, get_camera_name, open_camera
from core.video_window import VideoWindow


def list_available_effects():
    """Print all available effects"""
    effects = list_effects()

    if not effects:
        print("No effects found!")
        return

    print("\nAvailable Effects:")
    print("=" * 80)

    # Group by category
    by_category = {}
    for key, name, desc, category in effects:
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((key, name, desc))

    for category in sorted(by_category.keys()):
        print(f"\n{category.upper()}:")
        for key, name, desc in sorted(by_category[category]):
            print(f"  {key:30s} - {name}")
            if desc:
                print(f"  {' ' * 32}  {desc}")

    print("\n" + "=" * 80)
    print(f"Total: {len(effects)} effects")


def list_available_cameras():
    """Print all available cameras"""
    cameras = find_cameras()

    if not cameras:
        print("No cameras found!")
        return

    print("\nAvailable Cameras:")
    for idx in cameras:
        print(f"  {idx}: {get_camera_name(idx)}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified webcam filter system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List all available effects
  %(prog)s --effect misc/passthrough  Run the passthrough effect
  %(prog)s --effect seasonal/christmas --camera 1
        """
    )

    parser.add_argument('--effect', '-e', type=str, default='misc/passthrough',
                        help='Effect to run (default: misc/passthrough, use --list to see all)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available effects')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List all available cameras')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height (default: 720)')

    args = parser.parse_args()

    # Handle list commands
    if args.list:
        list_available_effects()
        return 0

    if args.list_cameras:
        list_available_cameras()
        return 0

    # Load the effect class (defaults to passthrough if not specified)
    try:
        effect_class = get_effect_class(args.effect)
    except KeyError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available effects")
        return 1

    # Open camera with specified or default resolution
    print(f"Opening camera {args.camera} at {args.width}x{args.height}...")
    cap = open_camera(args.camera, width=args.width, height=args.height)
    if cap is None:
        print(f"Error: Could not open camera {args.camera}")
        print("\nUse --list-cameras to see available cameras")
        return 1

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        cap.release()
        return 1

    height, width = frame.shape[:2]
    print(f"Camera resolution: {width}x{height}")

    # Debug: Check if frame is valid
    if frame is None or frame.size == 0:
        print("Error: Invalid frame from camera")
        cap.release()
        return 1

    # Create Tkinter root window (needed for UI effects)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create the effect instance
    print(f"Loading effect: {effect_class.get_name()}")
    if args.effect == 'misc/passthrough':
        print("(Using default passthrough effect - use --list to see other effects)")

    # Check if effect needs the root window (for UI effects)
    from core.base_effect import BaseUIEffect
    if issubclass(effect_class, BaseUIEffect):
        effect = effect_class(width, height, root)
    else:
        effect = effect_class(width, height)

    # Create video window
    video_window = VideoWindow(root, title=f"Webcam Filter - {effect_class.get_name()}",
                                width=width, height=height)

    # Force window to show and process events
    root.update()

    # Set up keyboard handler
    effect_enabled = True

    def handle_key(key):
        nonlocal effect_enabled
        if key == ' ':
            effect_enabled = not effect_enabled
            status = "ON" if effect_enabled else "OFF"
            print(f"Effect: {status}")
        elif key in ['q', 'esc']:
            video_window.is_open = False

    video_window.set_key_callback(handle_key)

    print(f"\nRunning {effect_class.get_name()}...")
    print("Controls:")
    print("  SPACE - Toggle effect on/off")
    print("  Q or ESC - Quit")

    # Main loop
    import time
    frame_count = 0
    try:
        while video_window.is_open:
            ret, frame = cap.read()
            if not ret:
                print("Error: Lost camera connection")
                break

            frame_count += 1
            if frame_count == 1:
                print(f"First frame received: shape={frame.shape}, dtype={frame.dtype}")

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Apply effect if enabled
            if effect_enabled:
                effect.update()
                frame = effect.draw(frame)

            if frame_count == 1:
                print(f"After effect: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

            # Display frame
            video_window.update_frame(frame)

            # Update Tkinter
            root.update_idletasks()
            root.update()

            # Small delay to prevent overwhelming the system
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("Cleaning up...")
        effect.cleanup()
        cap.release()
        root.quit()

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

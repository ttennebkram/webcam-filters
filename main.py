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

    parser.add_argument('--effect', '-e', type=str,
                        help='Effect to run (e.g., seasonal/christmas)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available effects')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List all available cameras')
    parser.add_argument('--camera', '-c', type=int, default=0,
                        help='Camera index (default: 0)')

    args = parser.parse_args()

    # Handle list commands
    if args.list:
        list_available_effects()
        return 0

    if args.list_cameras:
        list_available_cameras()
        return 0

    # Require an effect
    if not args.effect:
        parser.print_help()
        print("\nError: --effect is required (use --list to see available effects)")
        return 1

    # Load the effect class
    try:
        effect_class = get_effect_class(args.effect)
    except KeyError as e:
        print(f"Error: {e}")
        print("\nUse --list to see available effects")
        return 1

    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = open_camera(args.camera)
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

    # Create Tkinter root window (needed for UI effects)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Create the effect instance
    print(f"Loading effect: {effect_class.get_name()}")
    effect = effect_class(width, height, root)

    # Create video window
    video_window = VideoWindow(root, title=f"Webcam Filter - {effect_class.get_name()}",
                                width=width, height=height)

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
    try:
        while video_window.is_open:
            ret, frame = cap.read()
            if not ret:
                print("Error: Lost camera connection")
                break

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Apply effect if enabled
            if effect_enabled:
                effect.update()
                frame = effect.draw(frame)

            # Display frame
            video_window.update_frame(frame)

            # Update Tkinter
            root.update()

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

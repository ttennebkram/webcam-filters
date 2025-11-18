#!/usr/bin/env python3
"""
Unified webcam filter system.

Load and run any effect from the command line.
"""

import argparse
import sys
import tkinter as tk
from tkinter import ttk
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
        print(f"{category.upper()}:")
        for key, name, desc in sorted(by_category[category]):
            print(f"  {key:30s} - {name}")

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
  %(prog)s misc/passthrough           Run the passthrough effect
  %(prog)s seasonal/christmas --camera 1
        """
    )

    parser.add_argument('effect', nargs='?', type=str, default='misc/passthrough',
                        help='Effect to run (default: misc/passthrough, use --list to see all)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available effects')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List all available cameras')
    parser.add_argument('--camera', '-c', type=int, default=None,
                        help='Camera index (default: auto-detect highest index)')
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

    # Discover all available cameras
    cameras = find_cameras()
    if not cameras:
        print("Error: No cameras found!")
        return 1

    # Auto-detect camera if not specified (use highest index)
    camera_index = args.camera
    if camera_index is None:
        camera_index = max(cameras)  # Use highest index
        print(f"Auto-detected cameras: {cameras}")
        print(f"Using camera {camera_index} (highest index)")
    else:
        print(f"Available cameras: {cameras}")
        print(f"Using camera {camera_index} (specified)")

    # Open camera with specified or default resolution
    print(f"Opening camera {camera_index} at {args.width}x{args.height}...")
    cap = open_camera(camera_index, width=args.width, height=args.height)
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

    # Create global controls window
    global_controls = tk.Toplevel(root)
    global_controls.title("Webcam Filters - Global Controls")

    global_frame = ttk.Frame(global_controls, padding=10)
    global_frame.pack(fill=tk.BOTH, expand=True)

    # Top row with Camera, Resolution, and Flip in a grid layout
    top_row = ttk.Frame(global_frame)
    top_row.pack(fill='x', pady=(0, 10))

    # Column headers
    ttk.Label(top_row, text="Camera").grid(row=0, column=0, sticky='w', padx=5, pady=2)
    ttk.Label(top_row, text="Resolution").grid(row=0, column=1, sticky='w', padx=5, pady=2)
    ttk.Label(top_row, text="Mirror Flip").grid(row=0, column=2, sticky='w', padx=5, pady=2)

    # Shared state for camera/resolution changes
    camera_state = {
        'current_camera': camera_index,
        'current_width': width,
        'current_height': height,
        'needs_reopen': False
    }

    # Camera dropdown
    camera_var = tk.StringVar(value=f"{camera_index}: {get_camera_name(camera_index)}")
    camera_combo = ttk.Combobox(top_row, textvariable=camera_var, state='readonly', width=15)
    camera_combo['values'] = [f"{idx}: {get_camera_name(idx)}" for idx in cameras]
    camera_combo.current(cameras.index(camera_index))
    camera_combo.grid(row=1, column=0, sticky='w', padx=5, pady=2)

    def on_camera_change(event):
        new_camera = int(camera_var.get().split(':')[0])
        if new_camera != camera_state['current_camera']:
            print(f"Switching camera from {camera_state['current_camera']} to {new_camera}")
            camera_state['current_camera'] = new_camera
            camera_state['needs_reopen'] = True

    camera_combo.bind('<<ComboboxSelected>>', on_camera_change)

    # Resolution dropdown
    resolutions = [
        "640x480 (VGA)",
        "1024x768 (XGA)",
        "1280x720 (720p HD)",
        "1920x1080 (1080p Full HD)"
    ]

    current_res = f"{width}x{height}"
    resolution_var = tk.StringVar()
    found_match = False
    for res in resolutions:
        if current_res in res:
            resolution_var.set(res)
            found_match = True
            break
    if not found_match:
        resolution_var.set(f"{current_res} (Custom)")

    resolution_combo = ttk.Combobox(top_row, textvariable=resolution_var, state='readonly', width=20)
    resolution_combo['values'] = resolutions
    resolution_combo.grid(row=1, column=1, sticky='w', padx=5, pady=2)

    def on_resolution_change(event):
        try:
            res_str = resolution_var.get().split()[0]
            new_width, new_height = int(res_str.split('x')[0]), int(res_str.split('x')[1])
            if new_width != camera_state['current_width'] or new_height != camera_state['current_height']:
                print(f"Changing resolution from {camera_state['current_width']}x{camera_state['current_height']} to {new_width}x{new_height}")
                camera_state['current_width'] = new_width
                camera_state['current_height'] = new_height
                camera_state['needs_reopen'] = True
        except Exception as e:
            print(f"Error in resolution change: {e}")
            import traceback
            traceback.print_exc()

    resolution_combo.bind('<<ComboboxSelected>>', on_resolution_change)

    # Flip checkbox
    flip_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(top_row, text="Flip Left/Right", variable=flip_var).grid(row=1, column=2, sticky='w', padx=5, pady=2)

    # Show original checkbox (below the grid)
    show_original_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(global_frame, text="Show Original Image (disable all effects)",
                   variable=show_original_var).pack(anchor='w', pady=(10, 5))

    # Effect selection on its own row
    ttk.Label(global_frame, text="Effect (restarts program):").pack(anchor='w', pady=(5, 2))

    # Shared state for effect restart
    restart_info = {'should_restart': False, 'args': []}

    all_effects = list_effects()
    effect_keys = [key for key, name, desc, category in all_effects]

    effect_var = tk.StringVar(value=args.effect)
    effect_combo = ttk.Combobox(global_frame, textvariable=effect_var, state='readonly', width=40)
    effect_combo['values'] = effect_keys
    if args.effect in effect_keys:
        effect_combo.current(effect_keys.index(args.effect))
    effect_combo.pack(fill='x', pady=(0, 5))

    def on_effect_change(event):
        new_effect = effect_var.get()
        if new_effect != args.effect:
            restart_info['should_restart'] = True
            restart_info['args'] = [sys.executable, sys.argv[0], new_effect,
                                     '--camera', str(camera_state['current_camera']),
                                     '--width', str(camera_state['current_width']),
                                     '--height', str(camera_state['current_height'])]

    effect_combo.bind('<<ComboboxSelected>>', on_effect_change)

    ttk.Label(global_frame, text="Camera and resolution change instantly",
              font=('TkDefaultFont', 8, 'italic')).pack(anchor='w', pady=(5, 0))

    # Create the effect instance
    print(f"Loading effect: {effect_class.get_name()}")
    if args.effect == 'misc/passthrough':
        print("(Using default passthrough effect - use --list to see other effects)")

    # Check if effect needs the root window (for UI effects)
    from core.base_effect import BaseUIEffect
    if issubclass(effect_class, BaseUIEffect):
        effect = effect_class(width, height, root)

        # Create control panel window for UI effects
        if hasattr(effect, 'create_control_panel'):
            control_window = tk.Toplevel(root)
            control_window.title(f"{effect_class.get_name()} - Controls")
            control_window.geometry("450x600")  # Set larger default size
            control_window.minsize(400, 500)    # Prevent it from being too small
            control_panel = effect.create_control_panel(control_window)
            control_panel.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    else:
        effect = effect_class(width, height)

    # Create video window
    video_window = VideoWindow(root, title=f"Webcam Filter - {effect_class.get_name()}",
                                width=width, height=height)

    # Calculate and set window positions
    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Force update to get actual window dimensions
    root.update_idletasks()
    global_controls.update_idletasks()

    global_width = global_controls.winfo_reqwidth()
    global_height = global_controls.winfo_reqheight()

    # Window dimensions
    gap = 20  # Gap between windows horizontally
    vertical_gap = 20  # Gap between windows vertically
    video_width = width
    video_height = height

    # Calculate total width needed for global controls + video
    total_width = global_width + gap + video_width

    # Center the layout horizontally
    start_x = max(0, (screen_width - total_width) // 2)

    # Position global controls at top left
    global_x = start_x
    global_y = 0
    global_controls.geometry(f"{global_width}x{global_height}+{global_x}+{global_y}")

    # Position video window to the right of global controls
    video_x = global_x + global_width + gap
    video_y = 0
    video_window.window.geometry(f"{video_width}x{video_height}+{video_x}+{video_y}")

    # If there's a control window for the effect, position it below global controls
    if 'control_window' in locals():
        control_window.update_idletasks()
        control_width = control_window.winfo_reqwidth()
        control_height = control_window.winfo_reqheight()

        control_x = global_x
        control_y = global_y + global_height + vertical_gap
        control_window.geometry(f"{control_width}x{control_height}+{control_x}+{control_y}")

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
    import subprocess
    try:
        while video_window.is_open:
            # Check if effect change requested (requires restart)
            if restart_info['should_restart']:
                print("\nRestarting with new effect...")
                break

            # Check if camera/resolution needs to change
            if camera_state['needs_reopen']:
                print("Reopening camera with new settings...")
                cap.release()
                cap = open_camera(camera_state['current_camera'],
                                width=camera_state['current_width'],
                                height=camera_state['current_height'])
                if cap is None:
                    print(f"Error: Could not open camera {camera_state['current_camera']}")
                    break

                # Resize video window to match new resolution
                video_window.resize(camera_state['current_width'], camera_state['current_height'])

                camera_state['needs_reopen'] = False
                print(f"Camera reopened: {camera_state['current_camera']} at {camera_state['current_width']}x{camera_state['current_height']}")

            ret, frame = cap.read()
            if not ret:
                print("Error: Lost camera connection")
                break

            # Mirror the frame horizontally if enabled
            if flip_var.get():
                frame = cv2.flip(frame, 1)

            # Apply effect if enabled (both effect_enabled from spacebar AND not showing original)
            if effect_enabled and not show_original_var.get():
                effect.update()
                frame = effect.draw(frame)

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

        # If effect change was requested, launch new instance
        if restart_info['should_restart']:
            try:
                print(f"Launching: {' '.join(restart_info['args'])}")
                time.sleep(0.2)  # Brief pause for cleanup
                subprocess.Popen(restart_info['args'], start_new_session=True)
            except Exception as e:
                print(f"Error launching new instance: {e}")

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

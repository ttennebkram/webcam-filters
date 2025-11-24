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
import json
import os
import math

from effects import discover_effects, list_effects, get_effect_class
from core.camera import find_cameras, get_camera_name, open_camera
from core.video_window import VideoWindow

# Settings file path (in project directory)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


def load_settings():
    """Load settings from JSON file"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
    return {}


def save_settings(settings):
    """Save settings to JSON file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save settings: {e}")


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
    parser.add_argument('--edit-pipeline', type=str, default=None,
                        help='Pipeline key to load in Pipeline Builder (e.g., user_test1)')

    args = parser.parse_args()

    # Handle list commands
    if args.list:
        list_available_effects()
        return 0

    if args.list_cameras:
        list_available_cameras()
        return 0

    # Load saved settings early to get the saved effect
    saved_settings = load_settings()

    # Use saved effect if no effect was specified on command line
    effect_to_load = args.effect
    if args.effect == 'misc/passthrough' and 'effect' in saved_settings:
        effect_to_load = saved_settings['effect']
        print(f"Loading saved effect: {effect_to_load}")

    # Load the effect class (defaults to passthrough if not specified)
    try:
        effect_class = get_effect_class(effect_to_load)
    except KeyError as e:
        # If saved effect not found, fall back to passthrough
        if effect_to_load != args.effect:
            print(f"Warning: Saved effect '{effect_to_load}' not found, falling back to passthrough")
            effect_to_load = 'misc/passthrough'
            effect_class = get_effect_class(effect_to_load)
        else:
            print(f"Error: {e}")
            print("\nUse --list to see available effects")
            return 1

    # Discover all available cameras (needed for UI even if loading from file)
    cameras = find_cameras()
    if not cameras:
        cameras = [0]  # Fallback so UI doesn't break

    # Auto-detect camera index
    if args.camera is not None:
        camera_index = args.camera
    else:
        camera_index = max(cameras)

    # Check if we should load a file instead of camera
    initial_static_image = None
    cap = None
    file_loaded = False

    if saved_settings.get('input_source') == 'file' and saved_settings.get('file_path'):
        file_path = saved_settings['file_path']
        ext = file_path.lower().split('.')[-1]

        if ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
            # Static image
            print(f"Loading static image: {file_path}")
            img = cv2.imread(file_path)
            if img is not None:
                initial_static_image = img
                frame = img.copy()  # For validation
                height, width = img.shape[:2]
                file_loaded = True
                print(f"Static image resolution: {width}x{height}")
            else:
                print(f"Error: Could not load image {file_path}, falling back to camera")
                saved_settings['input_source'] = 'camera'
        else:
            # Video file
            print(f"Loading video file: {file_path}")
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    file_loaded = True
                    print(f"Video resolution: {width}x{height}")
                else:
                    print(f"Error: Could not read from video, falling back to camera")
                    cap.release()
                    cap = None
                    saved_settings['input_source'] = 'camera'
            else:
                print(f"Error: Could not open video {file_path}, falling back to camera")
                saved_settings['input_source'] = 'camera'

    # Open camera if not using file input (or as fallback)
    if not file_loaded:
        # Print camera info
        if args.camera is None:
            print(f"Auto-detected cameras: {cameras}")
            print(f"Using camera {camera_index} (highest index)")
        else:
            print(f"Available cameras: {cameras}")
            print(f"Using camera {camera_index} (specified)")

        # Open camera with specified or default resolution
        print(f"Opening camera {camera_index} at {args.width}x{args.height}...")
        cap = open_camera(camera_index, width=args.width, height=args.height)
        if cap is None:
            print(f"Error: Could not open camera {camera_index}")
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
        print("Error: Invalid frame from source")
        if cap is not None:
            cap.release()
        return 1

    # Create Tkinter root window (needed for UI effects)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Shared flag to trigger shutdown when any window is closed
    shutdown_flag = {'should_exit': False}

    def on_any_window_close():
        """Called when any window is closed - triggers complete shutdown"""
        shutdown_flag['should_exit'] = True

    # Create global controls window
    global_controls = tk.Toplevel(root)
    global_controls.withdraw()  # Hide until positioned
    global_controls.title("Global Filter Controls")
    global_controls.protocol("WM_DELETE_WINDOW", on_any_window_close)
    global_controls.configure(highlightthickness=0)

    global_frame = ttk.Frame(global_controls, padding=10)
    global_frame.pack(fill=tk.BOTH, expand=True)
    # Disable focus highlight
    global_frame.configure(takefocus=0)

    # Shared state for camera/resolution changes
    camera_state = {
        'current_camera': camera_index,
        'current_width': width,
        'current_height': height,
        'needs_reopen': False,
        'input_source': saved_settings.get('input_source', 'camera'),
        'file_path': saved_settings.get('file_path', ''),
        '_static_image': initial_static_image
    }

    # Input section header
    ttk.Label(global_frame, text="Input", font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 3))

    # Input source selection (radio buttons)
    input_source_var = tk.StringVar(value=saved_settings.get('input_source', 'camera'))
    camera_state['input_source'] = input_source_var.get()

    # Load saved file path
    saved_file_path = saved_settings.get('file_path', '')
    camera_state['file_path'] = saved_file_path

    # === Camera input row ===
    camera_row = ttk.Frame(global_frame)
    camera_row.pack(fill='x', pady=(0, 3))
    camera_row.configure(takefocus=0)

    # Radio button for camera
    camera_radio = ttk.Radiobutton(camera_row, text="Camera:", variable=input_source_var, value='camera')
    camera_radio.pack(side='left', padx=(0, 5))

    # Camera dropdown
    camera_var = tk.StringVar(value=f"{camera_index}: {get_camera_name(camera_index)}")
    camera_combo = ttk.Combobox(camera_row, textvariable=camera_var, state='readonly', width=12)
    camera_combo['values'] = [f"{idx}: {get_camera_name(idx)}" for idx in cameras]
    camera_combo.current(cameras.index(camera_index))
    camera_combo.pack(side='left', padx=(0, 10))

    def on_camera_change(event):
        new_camera = int(camera_var.get().split(':')[0])
        if new_camera != camera_state['current_camera']:
            print(f"Switching camera from {camera_state['current_camera']} to {new_camera}")
            camera_state['current_camera'] = new_camera
            camera_state['needs_reopen'] = True
        # Auto-select camera radio button
        input_source_var.set('camera')

    camera_combo.bind('<<ComboboxSelected>>', on_camera_change)

    # Resolution dropdown
    resolutions = [
        "640x480",
        "1024x768",
        "1280x720",
        "1920x1080"
    ]

    current_res = f"{width}x{height}"
    resolution_var = tk.StringVar()
    if current_res in resolutions:
        resolution_var.set(current_res)
    else:
        resolution_var.set(f"{current_res}")

    ttk.Label(camera_row, text="Resolution:").pack(side='left', padx=(0, 5))
    resolution_combo = ttk.Combobox(camera_row, textvariable=resolution_var, state='readonly', width=10)
    resolution_combo['values'] = resolutions
    resolution_combo.pack(side='left')

    def on_resolution_change(event):
        try:
            res_str = resolution_var.get()
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

    # === File input row ===
    file_row = ttk.Frame(global_frame)
    file_row.pack(fill='x', pady=(0, 3))
    file_row.configure(takefocus=0)

    # Radio button for file
    file_radio = ttk.Radiobutton(file_row, text="File:", variable=input_source_var, value='file')
    file_radio.pack(side='left', padx=(0, 5))

    # File path entry
    file_path_var = tk.StringVar(value=saved_file_path)
    file_entry = ttk.Entry(file_row, textvariable=file_path_var, width=30)
    file_entry.pack(side='left', padx=(0, 5), fill='x', expand=True)

    def on_file_path_change(*args):
        path = file_path_var.get()
        if path:
            camera_state['file_path'] = path
            camera_state['needs_reopen'] = True
            # Auto-select file radio button
            input_source_var.set('file')

    file_path_var.trace_add('write', on_file_path_change)

    # Browse button
    def browse_file():
        from tkinter import filedialog
        filetypes = [
            ("Image/Video files", "*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv"),
            ("Images", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("Videos", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            file_path_var.set(filepath)

    browse_btn = ttk.Button(file_row, text="...", width=3, command=browse_file)
    browse_btn.pack(side='left')

    # Track input source changes
    def on_input_source_change(*args):
        camera_state['input_source'] = input_source_var.get()
        camera_state['needs_reopen'] = True

    input_source_var.trace_add('write', on_input_source_change)

    # === Mirror flip row ===
    flip_row = ttk.Frame(global_frame)
    flip_row.pack(fill='x', pady=(0, 5))
    flip_row.configure(takefocus=0)

    flip_var = tk.BooleanVar(value=saved_settings.get('flip', True))
    ttk.Checkbutton(flip_row, text="Mirror Flip (Left/Right)", variable=flip_var).pack(side='left')

    # Effect selection on its own row
    ttk.Label(global_frame, text="Effect:").pack(anchor='w', pady=(5, 2))

    # Shared state for effect restart
    restart_info = {'should_restart': False, 'args': []}

    all_effects = list_effects()
    effect_keys = [key for key, name, desc, category in all_effects]

    # Add special entry for creating new user pipeline at end of opencv section
    # Find where opencv effects end
    new_pipeline_entry = "opencv/(new user pipeline)"
    opencv_end_idx = 0
    for i, key in enumerate(effect_keys):
        if key.startswith('opencv/'):
            opencv_end_idx = i + 1
    effect_keys.insert(opencv_end_idx, new_pipeline_entry)

    # Frame to hold effect dropdown and reload button
    effect_row = ttk.Frame(global_frame)
    effect_row.pack(fill='x', pady=(0, 3))
    effect_row.configure(takefocus=0)

    # Output section header
    ttk.Label(global_frame, text="Output", font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(5, 3))

    # Gain and Invert controls (like FFT ringing)
    ttk.Label(global_frame, text="Gain: 0.1x to 10x").pack(anchor='w', pady=(3, 0))

    # Container for slider and tick marks
    gain_container = ttk.Frame(global_frame)
    gain_container.pack(fill='x', pady=(0, 0))
    gain_container.configure(takefocus=0)

    gain_frame = ttk.Frame(gain_container)
    gain_frame.pack(fill='x')
    gain_frame.configure(takefocus=0)

    # Use logarithmic scale: slider goes from -1 to 1, maps to 10^slider (0.1 to 10)
    # Convert saved gain to log scale for slider
    saved_gain = saved_settings.get('gain', 1.0)
    initial_slider_value = math.log10(saved_gain) if saved_gain > 0 else 0

    gain_slider_var = tk.DoubleVar(value=initial_slider_value)
    gain_var = tk.DoubleVar(value=saved_gain)  # Actual gain value
    gain_display_var = tk.StringVar(value=f"{saved_gain:.2f}x")

    def on_gain_slider_change(log_value):
        """Convert logarithmic slider value to actual gain"""
        actual_gain = 10 ** float(log_value)
        gain_var.set(actual_gain)
        gain_display_var.set(f"{actual_gain:.2f}x")

    # Create slider (back to ttk.Scale for consistent styling)
    gain_slider = ttk.Scale(gain_frame, from_=-1, to=1, variable=gain_slider_var, orient='horizontal',
                           command=on_gain_slider_change)
    gain_slider.pack(side='left', fill='x', expand=True, padx=(0, 5))
    ttk.Label(gain_frame, textvariable=gain_display_var, width=7).pack(side='left')

    # Add tick marks below the slider
    # Use system background color for the canvas
    style = ttk.Style()
    bg_color = style.lookup('TFrame', 'background')
    tick_canvas = tk.Canvas(gain_container, height=8, bg=bg_color or 'SystemButtonFace', highlightthickness=0)
    tick_canvas.pack(fill='x', padx=(0, 52))  # Match slider padding + label width

    def draw_tick_marks():
        """Draw tick marks at logarithmic positions"""
        tick_canvas.delete('all')
        canvas_width = tick_canvas.winfo_width()
        if canvas_width <= 1:
            # Not yet rendered, try again later
            tick_canvas.after(100, draw_tick_marks)
            return

        # Get the actual slider widget dimensions
        try:
            slider_width = gain_slider.winfo_width()
            # ttk.Scale has internal padding, calculate based on actual slider position
            # The trough (the draggable area) is slightly inset from the widget edges
            # For ttk.Scale on macOS, this is typically about 9-10 pixels per side
            trough_padding = 9
            trough_width = slider_width - (2 * trough_padding)

            # Tick positions at all integer gain values
            # Small ticks at 0.1x through 0.9x, 2x through 9x, and 10x
            # Large tick at 1x (no gain)
            ticks = []

            # Add ticks for 0.1x to 0.9x
            for i in range(1, 10):
                gain = i / 10.0
                log_pos = math.log10(gain)
                ticks.append((log_pos, 3))

            # Large tick at 1x
            ticks.append((0.0, 6))

            # Add ticks for 2x to 10x
            for i in range(2, 11):
                gain = float(i)
                log_pos = math.log10(gain)
                ticks.append((log_pos, 3))

            for log_pos, height in ticks:
                # Map from -1..1 to the trough area
                x = int(trough_padding + (log_pos + 1) / 2 * trough_width)
                tick_canvas.create_line(x, 0, x, height, fill='gray40', width=1)
        except:
            # If we can't get dimensions yet, try again
            tick_canvas.after(100, draw_tick_marks)

    tick_canvas.bind('<Configure>', lambda e: draw_tick_marks())
    draw_tick_marks()

    # Invert checkbox
    invert_var = tk.BooleanVar(value=saved_settings.get('invert', False))
    ttk.Checkbutton(global_frame, text="Invert", variable=invert_var).pack(anchor='w', pady=(3, 3))

    # Show original checkbox (moved here from top, now in Output section)
    show_original_var = tk.BooleanVar(value=saved_settings.get('show_original', False))
    ttk.Checkbutton(global_frame, text="Show Original (no effects)",
                   variable=show_original_var).pack(anchor='w', pady=(0, 3))

    # Use the loaded effect (width removed to let it expand)
    effect_var = tk.StringVar(value=effect_to_load)
    effect_combo = ttk.Combobox(effect_row, textvariable=effect_var, state='readonly')
    effect_combo['values'] = effect_keys
    if effect_to_load in effect_keys:
        effect_combo.current(effect_keys.index(effect_to_load))
    effect_combo.pack(side='left', fill='x', expand=True)

    def on_effect_change(event):
        new_effect = effect_var.get()

        # Handle special "(new user pipeline)" entry
        if new_effect == new_pipeline_entry:
            # Switch to pipeline builder using hot-swap
            if switch_effect('opencv/pipeline_builder'):
                # Clear any existing pipeline to start fresh
                eff = effect_state['effect']
                if hasattr(eff, 'clear_pipeline'):
                    eff.clear_pipeline()
            return

        # Handle user pipelines - load in pipeline_builder instead of UserPipelineEffect
        if new_effect.startswith('opencv/user_'):
            # Extract pipeline key (e.g., "user_test6a" from "opencv/user_test6a")
            pipeline_key = new_effect.replace('opencv/', '')
            if switch_effect('opencv/pipeline_builder'):
                # Load the pipeline in the new pipeline builder instance
                eff = effect_state['effect']
                if hasattr(eff, '_load_pipeline_by_key'):
                    eff._load_pipeline_by_key(pipeline_key)
            return

        # Hot-swap to new effect if different from current
        if new_effect != effect_state['effect_key']:
            switch_effect(new_effect)

    def on_reload_click():
        """Reload the current effect by restarting the application"""
        restart_info['should_restart'] = True
        restart_info['args'] = [sys.executable, sys.argv[0], effect_state['effect_key'],
                                 '--camera', str(camera_state['current_camera']),
                                 '--width', str(camera_state['current_width']),
                                 '--height', str(camera_state['current_height'])]

        # If in Pipeline Builder with a loaded pipeline, reload as the user pipeline (view mode)
        current_effect = effect_state['effect']
        if effect_state['effect_key'] == 'opencv/pipeline_builder' and hasattr(current_effect, 'pipeline_name'):
            pipeline_name = current_effect.pipeline_name.get().strip()
            if pipeline_name:
                # Replace effect key with user pipeline to reload in view mode
                restart_info['args'][2] = f'opencv/user_{pipeline_name}'

    effect_combo.bind('<<ComboboxSelected>>', on_effect_change)

    # Reload button next to effect dropdown
    reload_button = ttk.Button(effect_row, text="Reload", command=on_reload_click, width=8)
    reload_button.pack(side='left', padx=(5, 0))

    ttk.Label(global_frame, text="Camera and resolution change instantly",
              font=('TkDefaultFont', 8, 'italic')).pack(anchor='w', pady=(5, 0))

    # Import base class for type checking
    from core.base_effect import BaseUIEffect

    # Mutable state for hot-swapping effects
    effect_state = {
        'effect': None,
        'effect_key': effect_to_load,
        'control_window': None,
        'canvas': None,
        'scrollbar': None,
        'scrollable_frame': None,
        'canvas_window': None
    }

    # Custom copy/paste functions for pipeline settings
    def copy_all_settings_text():
        """Copy all settings as human-readable text"""
        current_effect = effect_state['effect']
        if current_effect and hasattr(current_effect, '_copy_text'):
            current_effect._copy_text()
        else:
            print("Copy text not available for this effect")

    def copy_all_settings_json():
        """Copy all settings as JSON"""
        current_effect = effect_state['effect']
        if current_effect and hasattr(current_effect, '_copy_json'):
            current_effect._copy_json()
        else:
            print("Copy JSON not available for this effect")

    def paste_entire_pipeline():
        """Paste entire pipeline from clipboard (only in edit mode)

        Calls the existing _on_paste_key method which auto-detects format.
        """
        current_effect = effect_state['effect']
        if not current_effect:
            print("No effect loaded")
            return

        # Check if in edit mode
        if hasattr(current_effect, '_current_mode') and current_effect._current_mode != 'edit':
            print("Paste is only available in edit mode")
            return

        # Call the existing paste handler which auto-detects JSON vs text
        if hasattr(current_effect, '_on_paste_key'):
            current_effect._on_paste_key()
        else:
            print("Paste not available for this effect")

    def paste_pipeline_text():
        """Paste pipeline from clipboard as text format (only in edit mode)"""
        current_effect = effect_state['effect']
        if not current_effect:
            print("No effect loaded")
            return

        # Check if in edit mode
        if hasattr(current_effect, '_current_mode') and current_effect._current_mode != 'edit':
            print("Paste is only available in edit mode")
            return

        if hasattr(current_effect, '_paste_text'):
            current_effect._paste_text()
        else:
            print("Paste text not available for this effect")

    def paste_pipeline_json():
        """Paste pipeline from clipboard as JSON format (only in edit mode)"""
        current_effect = effect_state['effect']
        if not current_effect:
            print("No effect loaded")
            return

        # Check if in edit mode
        if hasattr(current_effect, '_current_mode') and current_effect._current_mode != 'edit':
            print("Paste is only available in edit mode")
            return

        if hasattr(current_effect, '_paste_json'):
            current_effect._paste_json()
        else:
            print("Paste JSON not available for this effect")

    def save_entire_pipeline():
        """Save the entire pipeline to disk and switch to view mode"""
        current_effect = effect_state['effect']
        if current_effect and hasattr(current_effect, '_on_edit_save_click'):
            # Use the same method as the Save All button - saves and switches to view mode
            current_effect._on_edit_save_click()
        elif current_effect and hasattr(current_effect, '_save_pipeline'):
            # Fallback to just saving
            current_effect._save_pipeline()
        else:
            print("Save not available for this effect")

    def new_pipeline():
        """Clear the current pipeline to start fresh"""
        current_effect = effect_state['effect']
        if current_effect and hasattr(current_effect, 'clear_pipeline'):
            current_effect.clear_pipeline()
        else:
            print("New pipeline not available for this effect")

    # Bind keyboard shortcuts (Cmd on macOS, Ctrl on Windows/Linux)
    root.bind('<Command-n>', lambda e: new_pipeline())
    root.bind('<Command-c>', lambda e: copy_all_settings_text())
    root.bind('<Command-j>', lambda e: copy_all_settings_json())
    root.bind('<Command-v>', lambda e: paste_entire_pipeline())
    root.bind('<Command-k>', lambda e: paste_pipeline_json())
    root.bind('<Command-s>', lambda e: save_entire_pipeline())
    root.bind('<Control-n>', lambda e: new_pipeline())
    root.bind('<Control-c>', lambda e: copy_all_settings_text())
    root.bind('<Control-j>', lambda e: copy_all_settings_json())
    root.bind('<Control-v>', lambda e: paste_entire_pipeline())
    root.bind('<Control-k>', lambda e: paste_pipeline_json())
    root.bind('<Control-s>', lambda e: save_entire_pipeline())

    # Create custom menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Detect platform for accelerator display
    import platform
    accel_mod = "Cmd" if platform.system() == "Darwin" else "Ctrl"

    def is_opencv_pipeline():
        """Check if current effect is an opencv pipeline effect"""
        current_effect = effect_state['effect']
        return current_effect and hasattr(current_effect, '_current_mode')

    def update_file_menu_state():
        """Enable/disable file menu items based on current effect"""
        if is_opencv_pipeline():
            file_menu.entryconfig(0, state='normal')  # New Pipeline
            file_menu.entryconfig(1, state='normal')  # Save Entire Pipeline
        else:
            file_menu.entryconfig(0, state='disabled')
            file_menu.entryconfig(1, state='disabled')

    # File menu
    file_menu = tk.Menu(menubar, tearoff=0, postcommand=update_file_menu_state)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New Pipeline", command=new_pipeline, accelerator=f"{accel_mod}+N")
    file_menu.add_command(label="Save Entire Pipeline", command=save_entire_pipeline, accelerator=f"{accel_mod}+S")

    def edit_entire_pipeline():
        """Enter edit mode for the pipeline"""
        current_effect = effect_state['effect']
        if current_effect and hasattr(current_effect, '_on_edit_save_click'):
            # Only enter edit mode if currently in view mode
            if hasattr(current_effect, '_current_mode') and current_effect._current_mode == 'view':
                current_effect._on_edit_save_click()
        else:
            print("Edit not available for this effect")

    def update_edit_menu_state():
        """Update edit menu items based on current effect and mode"""
        if is_opencv_pipeline():
            current_effect = effect_state['effect']
            # Enable all items
            edit_menu.entryconfig(0, state='normal')  # Edit Entire Pipeline
            edit_menu.entryconfig(1, state='normal')  # Copy as Text
            edit_menu.entryconfig(2, state='normal')  # Copy as JSON
            # Paste only enabled in edit mode
            if current_effect._current_mode == 'edit':
                edit_menu.entryconfig(3, state='normal')
            else:
                edit_menu.entryconfig(3, state='disabled')
        else:
            # Disable all items
            edit_menu.entryconfig(0, state='disabled')
            edit_menu.entryconfig(1, state='disabled')
            edit_menu.entryconfig(2, state='disabled')
            edit_menu.entryconfig(3, state='disabled')

    # Edit menu with our custom items
    edit_menu = tk.Menu(menubar, tearoff=0, postcommand=update_edit_menu_state)
    menubar.add_cascade(label="Edit", menu=edit_menu)

    edit_menu.add_command(label="Edit Entire Pipeline", command=edit_entire_pipeline, accelerator="↵")
    edit_menu.add_command(label="Copy Entire Pipeline as Text", command=copy_all_settings_text, accelerator=f"{accel_mod}+C")
    edit_menu.add_command(label="Copy Entire Pipeline as JSON", command=copy_all_settings_json, accelerator=f"{accel_mod}+J")
    edit_menu.add_command(label="Paste Entire Pipeline", command=paste_entire_pipeline, accelerator=f"{accel_mod}+V", state='disabled')

    def create_effect_instance(effect_key, effect_cls, w, h):
        """Create an effect instance with proper initialization"""
        if issubclass(effect_cls, BaseUIEffect):
            eff = effect_cls(w, h, root)
            # Pass edit-pipeline argument to Pipeline Builder
            if args.edit_pipeline and hasattr(eff, '_pipeline_to_load'):
                eff._pipeline_to_load = args.edit_pipeline
        else:
            eff = effect_cls(w, h)
        return eff

    def create_control_window_for_effect(eff, effect_cls):
        """Create a control panel window for a UI effect

        Returns:
            Tuple of (ctrl_window, canvas, scrollbar, scrollable_frame, canvas_window, ctrl_height)
        """
        if not hasattr(eff, 'create_control_panel'):
            return None, None, None, None, None, None

        ctrl_window = tk.Toplevel(root)
        ctrl_window.withdraw()  # Hide until positioned
        # Use custom title if effect instance has one, otherwise default
        if hasattr(eff, 'get_control_title'):
            ctrl_window.title(eff.get_control_title())
            # Store window reference for dynamic title updates
            if hasattr(eff, '_control_window'):
                eff._control_window = ctrl_window
        else:
            ctrl_window.title(f"{effect_cls.get_name()} - Controls")

        # Calculate available height based on screen size
        temp_screen_height = root.winfo_screenheight()
        # Leave room for menu bar and dock (roughly 150px)
        max_control_height = temp_screen_height - 150

        # Check if effect has a preferred height from config
        preferred_height = 800  # Default
        if hasattr(eff, 'get_preferred_window_height'):
            preferred_height = eff.get_preferred_window_height()
            print(f"DEBUG: Effect preferred height: {preferred_height}")

        # Use preferred height but cap at screen max
        ctrl_height = min(preferred_height, max_control_height)
        print(f"DEBUG: Control window height: {ctrl_height} (max: {max_control_height})")

        ctrl_window.geometry(f"700x{ctrl_height}")
        ctrl_window.minsize(700, 400)
        ctrl_window.protocol("WM_DELETE_WINDOW", on_any_window_close)

        # Create scrollable container for control panel
        canvas = tk.Canvas(ctrl_window, bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(ctrl_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        def _on_frame_configure(event):
            # Save current scroll position
            current_pos = canvas.yview()
            # Update scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Restore scroll position (or scroll to show bottom if we were near it)
            if current_pos[1] > 0.95:  # If near bottom, stay at bottom
                canvas.yview_moveto(1.0)
            else:
                canvas.yview_moveto(current_pos[0])

        scrollable_frame.bind("<Configure>", _on_frame_configure)

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make the scrollable frame expand to canvas width
        def _configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _configure_canvas_width)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            # macOS uses different delta scaling than Windows
            import platform
            if platform.system() == 'Darwin':
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mouse wheel events - macOS needs both MouseWheel and trackpad gestures
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # macOS trackpad scrolling
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        control_panel = eff.create_control_panel(scrollable_frame)
        control_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        return ctrl_window, canvas, scrollbar, scrollable_frame, canvas_window, ctrl_height

    def switch_effect(new_effect_key):
        """Hot-swap to a new effect without restarting the application"""
        nonlocal video_window

        print(f"\nSwitching to effect: {new_effect_key}")

        # Get the new effect class
        try:
            new_effect_class = get_effect_class(new_effect_key)
        except KeyError as e:
            print(f"Error: Could not load effect '{new_effect_key}': {e}")
            return False

        # Cleanup old effect
        if effect_state['effect'] is not None:
            print("Cleaning up old effect...")
            effect_state['effect'].cleanup()

        # Destroy old control window if it exists
        if effect_state['control_window'] is not None:
            effect_state['control_window'].destroy()
            effect_state['control_window'] = None
            effect_state['canvas'] = None
            effect_state['scrollbar'] = None
            effect_state['scrollable_frame'] = None
            effect_state['canvas_window'] = None

        # Get current frame dimensions
        current_width = camera_state['current_width']
        current_height = camera_state['current_height']

        # Create new effect instance
        print(f"Loading effect: {new_effect_class.get_name()}")
        new_effect = create_effect_instance(new_effect_key, new_effect_class, current_width, current_height)

        # Create control window if needed
        if issubclass(new_effect_class, BaseUIEffect):
            ctrl_window, canvas, scrollbar, scrollable_frame, canvas_window, ctrl_height = \
                create_control_window_for_effect(new_effect, new_effect_class)
            effect_state['control_window'] = ctrl_window
            effect_state['canvas'] = canvas
            effect_state['scrollbar'] = scrollbar
            effect_state['scrollable_frame'] = scrollable_frame
            effect_state['canvas_window'] = canvas_window

            # Position control window if it was created
            if ctrl_window is not None:
                root.update_idletasks()
                ctrl_window.update_idletasks()

                # Get global controls position for reference
                global_x = global_controls.winfo_x()
                global_y = global_controls.winfo_y()
                global_height = global_controls.winfo_height()

                # Position below global controls
                # Use reqwidth for width, but use the height that was set in geometry
                ctrl_width = ctrl_window.winfo_reqwidth()
                ctrl_x = global_x
                ctrl_y = global_y + global_height + 80  # vertical gap
                ctrl_window.geometry(f"{ctrl_width}x{ctrl_height}+{ctrl_x}+{ctrl_y}")
                ctrl_window.deiconify()  # Show now that it's positioned

        # Update state
        effect_state['effect'] = new_effect
        effect_state['effect_key'] = new_effect_key

        # Update video window title
        video_window.window.title(f"Webcam Filter - {new_effect_class.get_name()}")

        # Position visualization windows if they exist
        if hasattr(new_effect, 'viz_window') and new_effect.viz_window is not None:
            root.update_idletasks()
            new_effect.viz_window.update_idletasks()
            viz_width = new_effect.viz_window.winfo_reqwidth()
            viz_height = new_effect.viz_window.winfo_reqheight()

            video_x = video_window.window.winfo_x()
            video_y = video_window.window.winfo_y()
            video_height = video_window.window.winfo_height()

            viz_x = video_x
            viz_y = video_y + video_height + 80
            new_effect.viz_window.geometry(f"{viz_width}x{viz_height}+{viz_x}+{viz_y}")
            new_effect.viz_window.deiconify()  # Show now that it's positioned

        # Position difference window if it exists
        if hasattr(new_effect, 'diff_window') and new_effect.diff_window is not None:
            video_x = video_window.window.winfo_x()
            video_y = video_window.window.winfo_y()
            video_width_actual = video_window.window.winfo_width()

            diff_width = current_width
            diff_height = current_height
            diff_x = video_x + int(video_width_actual * 0.8)
            diff_y = video_y
            new_effect.diff_window.geometry(f"{diff_width}x{diff_height}+{diff_x}+{diff_y}")
            new_effect.diff_window.deiconify()  # Show now that it's positioned

        print(f"Effect switched to: {new_effect_class.get_name()}")
        return True

    # Handle user pipelines - redirect to pipeline_builder
    if effect_to_load.startswith('opencv/user_'):
        pipeline_key = effect_to_load.replace('opencv/', '')
        print(f"Loading user pipeline '{pipeline_key}' via Pipeline Builder...")
        effect_to_load = 'opencv/pipeline_builder'
        effect_class = get_effect_class(effect_to_load)
        # Set the pipeline to load after creation
        args.edit_pipeline = pipeline_key

    # Create the initial effect instance
    print(f"Loading effect: {effect_class.get_name()}")
    if effect_to_load == 'misc/passthrough' and args.effect == 'misc/passthrough':
        print("(Using default passthrough effect - use --list to see other effects)")

    effect = create_effect_instance(effect_to_load, effect_class, width, height)
    effect_state['effect'] = effect
    effect_state['effect_key'] = effect_to_load

    # Create control panel window for UI effects
    initial_ctrl_height = None
    if issubclass(effect_class, BaseUIEffect):
        ctrl_window, canvas, scrollbar, scrollable_frame, canvas_window, initial_ctrl_height = \
            create_control_window_for_effect(effect, effect_class)
        effect_state['control_window'] = ctrl_window
        effect_state['canvas'] = canvas
        effect_state['scrollbar'] = scrollbar
        effect_state['scrollable_frame'] = scrollable_frame
        effect_state['canvas_window'] = canvas_window

    # Create video window
    video_window = VideoWindow(root, title=f"Webcam Filter - {effect_class.get_name()}",
                                width=width, height=height, on_close_callback=on_any_window_close)

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
    gap = 160  # Gap between windows horizontally
    vertical_gap = 80  # Gap between windows vertically (increased to prevent overlap)
    video_width = width
    video_height = height

    # Determine the width of the left column (global controls or FFT control window)
    # If there's a control window, it will be below global controls and might be wider
    left_column_width = global_width
    if effect_state['control_window'] is not None:
        effect_state['control_window'].update_idletasks()
        control_width = effect_state['control_window'].winfo_reqwidth()
        # Use the maximum of global controls width and effect control width
        left_column_width = max(global_width, control_width)

    # Calculate total width needed for left column + video
    total_width = left_column_width + gap + video_width

    # Center the layout horizontally
    start_x = max(0, (screen_width - total_width) // 2)

    # Position global controls at top left
    global_x = start_x
    global_y = 0
    global_controls.geometry(f"{global_width}x{global_height}+{global_x}+{global_y}")

    # Position video window to the right of the left column (accounting for widest window)
    video_x = start_x + left_column_width + gap
    video_y = 0
    video_window.window.geometry(f"{video_width}x{video_height}+{video_x}+{video_y}")

    # If there's a control window for the effect, position it below global controls
    if effect_state['control_window'] is not None:
        ctrl_win = effect_state['control_window']
        ctrl_win.update_idletasks()
        ctrl_width = ctrl_win.winfo_reqwidth()
        # Use the height that was set when creating the window
        ctrl_height = initial_ctrl_height if initial_ctrl_height else ctrl_win.winfo_height()
        control_x = global_x
        control_y = global_y + global_height + vertical_gap
        ctrl_win.geometry(f"{ctrl_width}x{ctrl_height}+{control_x}+{control_y}")

    # If there's a visualization window (for effects like FFT), position it below the video window
    initial_effect = effect_state['effect']
    if hasattr(initial_effect, 'viz_window') and initial_effect.viz_window is not None:
        initial_effect.viz_window.update_idletasks()
        viz_width = initial_effect.viz_window.winfo_reqwidth()
        viz_height = initial_effect.viz_window.winfo_reqheight()

        # Position aligned with video window horizontally, below it vertically
        viz_x = video_x  # Same left edge as video window
        viz_y = video_y + video_height + vertical_gap  # Below video window

        initial_effect.viz_window.geometry(f"{viz_width}x{viz_height}+{viz_x}+{viz_y}")

    # If there's a difference window (for FFT effect), position it to the right of video window
    if hasattr(initial_effect, 'diff_window') and initial_effect.diff_window is not None:
        # Use the actual video dimensions for the diff window size
        diff_width = video_width
        diff_height = video_height

        # Position to the right of the video window, with slight overlap
        # Start 80% across the video window width
        diff_x = video_x + int(video_width * 0.8)
        diff_y = video_y  # Same vertical position as video window

        initial_effect.diff_window.geometry(f"{diff_width}x{diff_height}+{diff_x}+{diff_y}")

    # Show all windows now that they're positioned correctly
    global_controls.deiconify()
    video_window.window.deiconify()
    if effect_state['control_window'] is not None:
        effect_state['control_window'].deiconify()
    if hasattr(initial_effect, 'viz_window') and initial_effect.viz_window is not None:
        initial_effect.viz_window.deiconify()
    if hasattr(initial_effect, 'diff_window') and initial_effect.diff_window is not None:
        initial_effect.diff_window.deiconify()

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
            shutdown_flag['should_exit'] = True

    video_window.set_key_callback(handle_key)

    # Handle edit pipeline event from user pipelines
    def on_edit_pipeline(event):
        # Get pipeline key from the effect
        current_effect = effect_state['effect']
        pipeline_key = getattr(current_effect, '_pipeline_key', None)
        if pipeline_key:
            print(f"\nSwitching to Pipeline Builder to edit '{pipeline_key}'...")
            # Switch to pipeline builder and set it to load the pipeline
            if switch_effect('opencv/pipeline_builder'):
                # Set the pipeline to load after switching
                new_effect = effect_state['effect']
                if hasattr(new_effect, '_load_pipeline_by_key'):
                    new_effect._load_pipeline_by_key(pipeline_key)
                elif hasattr(new_effect, 'pipeline_name'):
                    # Fallback: just set the name
                    new_effect.pipeline_name.set(pipeline_key.replace('user_', ''))

    root.bind('<<EditPipeline>>', on_edit_pipeline)

    # Handle pipeline saved event - refresh the effect dropdown
    def on_pipeline_saved(event):
        # Re-discover all effects including new pipelines
        updated_effects = list_effects()
        updated_keys = [key for key, name, desc, category in updated_effects]

        # Add special entry for creating new user pipeline
        opencv_end_idx = 0
        for i, key in enumerate(updated_keys):
            if key.startswith('opencv/'):
                opencv_end_idx = i + 1
        updated_keys.insert(opencv_end_idx, new_pipeline_entry)

        # Update the combobox values
        effect_combo['values'] = updated_keys
        print("Effect list refreshed with new pipeline")

    root.bind('<<PipelineSaved>>', on_pipeline_saved)

    # Handle run pipeline event - switch from Pipeline Builder to the saved pipeline
    def on_run_pipeline(event):
        current_effect = effect_state['effect']
        if hasattr(current_effect, 'pipeline_name'):
            name = current_effect.pipeline_name.get().strip()
            if name:
                pipeline_key = f"opencv/user_{name}"
                print(f"\nSwitching to run pipeline '{name}'...")
                switch_effect(pipeline_key)

    root.bind('<<RunPipeline>>', on_run_pipeline)

    print(f"\nRunning {effect_class.get_name()}...")
    print("Controls:")
    print("  SPACE - Toggle effect on/off")
    print("  Q or ESC - Quit")

    # Main loop
    import time
    import subprocess
    try:
        while video_window.is_open and not shutdown_flag['should_exit']:
            # Check if effect change requested (requires restart)
            if restart_info['should_restart']:
                print("\nRestarting with new effect...")
                break

            # Check if shutdown requested
            if shutdown_flag['should_exit']:
                print("\nShutting down...")
                break

            # Check if camera/resolution needs to change
            if camera_state['needs_reopen']:
                if camera_state['input_source'] == 'camera':
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
                    camera_state['_static_image'] = None
                    print(f"Camera reopened: {camera_state['current_camera']} at {camera_state['current_width']}x{camera_state['current_height']}")

                elif camera_state['input_source'] == 'file' and camera_state['file_path']:
                    file_path = camera_state['file_path']
                    print(f"Opening file: {file_path}")

                    # Check if it's an image or video
                    ext = file_path.lower().split('.')[-1]
                    if ext in ['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff']:
                        # Static image
                        img = cv2.imread(file_path)
                        if img is not None:
                            camera_state['_static_image'] = img
                            h, w = img.shape[:2]
                            video_window.resize(w, h)
                            print(f"Loaded static image: {w}x{h}")
                        else:
                            print(f"Error: Could not load image {file_path}")
                    else:
                        # Video file
                        cap.release()
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            video_window.resize(w, h)
                            camera_state['_static_image'] = None
                            print(f"Loaded video: {w}x{h}")
                        else:
                            print(f"Error: Could not open video {file_path}")

                    camera_state['needs_reopen'] = False

            # Read frame from appropriate source
            if camera_state.get('_static_image') is not None:
                # Use static image
                frame = camera_state['_static_image'].copy()
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    # For video files, loop back to start
                    if camera_state['input_source'] == 'file':
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret:
                        print("Error: Lost camera/video connection")
                        break

            # Mirror the frame horizontally if enabled
            if flip_var.get():
                frame = cv2.flip(frame, 1)

            # Apply effect if enabled (both effect_enabled from spacebar AND not showing original)
            if effect_enabled and not show_original_var.get():
                current_effect = effect_state['effect']
                current_effect.update()
                frame = current_effect.draw(frame)

                # Apply global gain if not 1.0
                current_gain = gain_var.get()
                if current_gain != 1.0:
                    frame = cv2.convertScaleAbs(frame, alpha=current_gain)

                # Apply global invert if enabled
                if invert_var.get():
                    frame = 255 - frame

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
        # Save settings before cleanup
        current_settings = {
            'effect': effect_var.get(),
            'flip': flip_var.get(),
            'show_original': show_original_var.get(),
            'gain': gain_var.get(),
            'invert': invert_var.get(),
            'input_source': input_source_var.get(),
            'file_path': file_path_var.get()
        }
        save_settings(current_settings)

        # Cleanup
        print("Cleaning up...")
        if effect_state['effect'] is not None:
            effect_state['effect'].cleanup()
        if cap is not None:
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

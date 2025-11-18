"""
Tkinter-based video display window.

Uses PIL/ImageTk for accurate rendering without OpenCV's auto-contrast issues.
"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2


class VideoWindow:
    """Tkinter window for displaying video using PIL/ImageTk"""

    def __init__(self, root, title="Video", width=640, height=480):
        """Create a video display window

        Args:
            root: Parent Tkinter root window
            title: Window title
            width: Initial window width
            height: Initial window height
        """
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title(title)

        # Use Canvas for direct pixel control
        self.canvas = tk.Canvas(self.window, width=width, height=height, highlightthickness=0)
        self.canvas.pack()

        # Store current photo reference to prevent garbage collection
        self.current_photo = None
        self.canvas_image_id = None

        # Position window
        self.window.geometry(f"{width}x{height}+0+0")

        # Keyboard handler callback
        self.on_key_callback = None

        # Bind keyboard events
        self.window.bind('<space>', lambda e: self._handle_key(' '))
        self.window.bind('<q>', lambda e: self._handle_key('q'))
        self.window.bind('<Escape>', lambda e: self._handle_key('esc'))

        # Flag to check if window is open
        self.is_open = True
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _handle_key(self, key):
        """Handle keyboard input"""
        if self.on_key_callback:
            self.on_key_callback(key)

    def _on_close(self):
        """Handle window close"""
        self.is_open = False
        self.window.destroy()

    def update_frame(self, frame_bgr):
        """Update the displayed frame

        Args:
            frame_bgr: Frame in BGR format (OpenCV standard)
        """
        if not self.is_open:
            return

        try:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=image)

            # Update canvas
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            else:
                self.canvas.itemconfig(self.canvas_image_id, image=photo)

            # Keep reference to prevent garbage collection
            self.current_photo = photo

        except Exception as e:
            print(f"Error updating frame: {e}")
            import traceback
            traceback.print_exc()

    def set_key_callback(self, callback):
        """Set callback function for keyboard events

        Args:
            callback: Function that takes a key string as argument
        """
        self.on_key_callback = callback

    def set_position(self, x, y):
        """Set window position

        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.window.geometry(f"+{x}+{y}")

    def resize(self, width, height):
        """Resize the window and canvas

        Args:
            width: New width
            height: New height
        """
        self.canvas.config(width=width, height=height)
        # Get current position
        geometry = self.window.geometry()
        if '+' in geometry:
            pos = geometry.split('+', 1)[1]
            self.window.geometry(f"{width}x{height}+{pos}")
        else:
            self.window.geometry(f"{width}x{height}")

"""
Tkinter-based video display window.

Uses PIL/ImageTk for accurate rendering without OpenCV's auto-contrast issues.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


class VideoWindow:
    """Tkinter window for displaying video using PIL/ImageTk"""

    def __init__(self, root, title="Video", width=640, height=480, on_close_callback=None):
        """Create a video display window

        Args:
            root: Parent Tkinter root window
            title: Window title
            width: Initial window width
            height: Initial window height
            on_close_callback: Optional callback function to call when window is closed
        """
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title(title)

        # Store image dimensions
        self.image_width = width
        self.image_height = height

        # Create main frame to hold canvas and scrollbars
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.main_frame, highlightthickness=0, bg='black')

        # Vertical scrollbar
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Configure canvas scrolling
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Store current photo reference to prevent garbage collection
        self.current_photo = None
        self.canvas_image_id = None

        # Position window - set initial size with some reasonable maximum
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Window size: use image size but cap at 80% of screen
        window_width = min(width, int(screen_width * 0.8))
        window_height = min(height, int(screen_height * 0.8))

        self.window.geometry(f"{window_width}x{window_height}+0+0")

        # Set scroll region to image size
        self.canvas.configure(scrollregion=(0, 0, width, height))

        # Enable mouse wheel scrolling
        self._bind_mousewheel()

        # Keyboard handler callback
        self.on_key_callback = None

        # Close callback
        self.on_close_callback = on_close_callback

        # Bind keyboard events
        self.window.bind('<space>', lambda e: self._handle_key(' '))
        self.window.bind('<q>', lambda e: self._handle_key('q'))
        self.window.bind('<Escape>', lambda e: self._handle_key('esc'))

        # Flag to check if window is open
        self.is_open = True
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _bind_mousewheel(self):
        """Bind mouse wheel events for scrolling"""
        # macOS uses MouseWheel event differently
        self.canvas.bind('<MouseWheel>', self._on_mousewheel_y)
        self.canvas.bind('<Shift-MouseWheel>', self._on_mousewheel_x)

        # Linux uses Button-4/5 for scroll
        self.canvas.bind('<Button-4>', lambda e: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind('<Button-5>', lambda e: self.canvas.yview_scroll(1, 'units'))
        self.canvas.bind('<Shift-Button-4>', lambda e: self.canvas.xview_scroll(-1, 'units'))
        self.canvas.bind('<Shift-Button-5>', lambda e: self.canvas.xview_scroll(1, 'units'))

    def _on_mousewheel_y(self, event):
        """Handle vertical mouse wheel scroll"""
        # macOS delta is inverted and larger
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _on_mousewheel_x(self, event):
        """Handle horizontal mouse wheel scroll (with Shift)"""
        self.canvas.xview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _handle_key(self, key):
        """Handle keyboard input"""
        if self.on_key_callback:
            self.on_key_callback(key)

    def _on_close(self):
        """Handle window close"""
        self.is_open = False
        if self.on_close_callback:
            self.on_close_callback()
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

            # Check if image size changed and update scroll region
            img_width, img_height = image.size
            if img_width != self.image_width or img_height != self.image_height:
                self.image_width = img_width
                self.image_height = img_height
                self.canvas.configure(scrollregion=(0, 0, img_width, img_height))

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
        """Resize for new image dimensions

        Args:
            width: New image width
            height: New image height
        """
        # Update stored image dimensions
        self.image_width = width
        self.image_height = height

        # Update scroll region to match image size
        self.canvas.configure(scrollregion=(0, 0, width, height))

        # Resize window, capping at 80% of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        window_width = min(width, int(screen_width * 0.8))
        window_height = min(height, int(screen_height * 0.8))

        # Get current position
        geometry = self.window.geometry()
        if '+' in geometry:
            pos = geometry.split('+', 1)[1]
            self.window.geometry(f"{window_width}x{window_height}+{pos}")
        else:
            self.window.geometry(f"{window_width}x{window_height}")

        # Reset scroll position to top-left
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

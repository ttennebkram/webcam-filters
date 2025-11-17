"""
Base classes for webcam filter effects.

All effects should inherit from BaseEffect and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class BaseEffect(ABC):
    """Abstract base class for all webcam filter effects"""

    def __init__(self, width: int, height: int):
        """Initialize the effect with frame dimensions

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Return the display name of this effect

        Returns:
            Human-readable name (e.g., "Christmas Garland")
        """
        pass

    @classmethod
    def get_description(cls) -> str:
        """Return a brief description of this effect

        Returns:
            One-line description of what the effect does
        """
        return ""

    @classmethod
    def get_category(cls) -> str:
        """Return the category this effect belongs to

        Returns:
            Category name (e.g., "seasonal", "matrix", "lines", "refraction", "misc")
        """
        return "misc"

    def update(self):
        """Update animation state

        Called once per frame before draw(). Use this to update positions,
        timers, and other animation state.
        """
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply the effect to a frame

        Args:
            frame: Input frame (BGR format, numpy array)
            face_mask: Optional face detection mask from MediaPipe

        Returns:
            Processed frame (BGR format, numpy array)
        """
        pass

    def cleanup(self):
        """Clean up resources when effect is no longer needed

        Override this if your effect needs to release resources,
        close windows, stop threads, etc.
        """
        pass


class BaseUIEffect(BaseEffect):
    """Base class for effects that have a control panel UI

    Effects that need interactive controls should inherit from this
    instead of BaseEffect.
    """

    def __init__(self, width: int, height: int, root_window=None):
        """Initialize effect with optional root window for UI

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            root_window: Tkinter root window (if UI is needed)
        """
        super().__init__(width, height)
        self.root_window = root_window
        self.control_panel = None

    @abstractmethod
    def create_control_panel(self, parent):
        """Create the Tkinter control panel for this effect

        Args:
            parent: Parent Tkinter widget

        Returns:
            The control panel widget
        """
        pass

    def get_control_panel(self):
        """Get the control panel if it exists

        Returns:
            Control panel widget or None
        """
        return self.control_panel


class BaseVisualizationEffect(BaseUIEffect):
    """Base class for effects that have external visualization windows

    Effects that show graphs, spectrograms, or other visualizations
    should inherit from this.
    """

    def __init__(self, width: int, height: int, root_window=None):
        """Initialize effect with visualization support

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            root_window: Tkinter root window (if UI/viz is needed)
        """
        super().__init__(width, height, root_window)
        self.viz_window = None

    @abstractmethod
    def create_visualization_window(self, parent):
        """Create the visualization window for this effect

        Args:
            parent: Parent Tkinter widget

        Returns:
            The visualization window
        """
        pass

    def update_visualization(self, *args, **kwargs):
        """Update the visualization with new data

        Override this to update graphs, plots, etc.
        """
        pass

    def get_visualization_window(self):
        """Get the visualization window if it exists

        Returns:
            Visualization window or None
        """
        return self.viz_window

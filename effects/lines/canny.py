"""
Canny Edge Detection Effect

Simple Canny edge detector with adjustable parameters for clean edge visualization.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


# Configuration constants
DEFAULT_BLUR_KERNEL = 3  # Default blur kernel size for Canny (must be odd)
DEFAULT_THRESHOLD1 = 25  # Default Canny lower threshold
DEFAULT_THRESHOLD2 = 7   # Default Canny upper threshold
DEFAULT_APERTURE_SIZE = 3  # Default Sobel kernel size (3, 5, or 7)
DEFAULT_L2_GRADIENT = True  # Default gradient calculation method (True = L2, False = L1)


class LinesCanny(BaseEffect):
    """Simple Canny edge detector with adjustable parameters"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Canny parameters with custom defaults
        self.blur_kernel = DEFAULT_BLUR_KERNEL
        self.threshold1 = DEFAULT_THRESHOLD1
        self.threshold2 = DEFAULT_THRESHOLD2
        self.aperture_size = DEFAULT_APERTURE_SIZE
        self.l2_gradient = DEFAULT_L2_GRADIENT

    @classmethod
    def get_name(cls) -> str:
        return "Canny Edges"

    @classmethod
    def get_description(cls) -> str:
        return "Simple Canny edge detection with adjustable parameters"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply Canny edge detection and return edges as grayscale image"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur (only if blur_kernel > 1)
        if self.blur_kernel > 1:
            blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        else:
            blurred = gray

        # Apply Canny edge detection with all parameters
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2,
                         apertureSize=self.aperture_size, L2gradient=self.l2_gradient)

        # Convert to 3-channel for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return result

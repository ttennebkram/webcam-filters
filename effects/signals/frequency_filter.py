"""
Frequency Filter Effect

Extracts high-frequency details from the webcam feed using a high-pass filter.
Shows fine details and edges by removing low-frequency information.
"""

import cv2
import numpy as np
from core.base_effect import BaseEffect


# Configuration constants
DEFAULT_BLUR_KERNEL = 95  # Default blur kernel size for high-pass filter (must be odd)


class FrequencyFilterEffect(BaseEffect):
    """High-pass filter to extract fine details from image"""

    def __init__(self, width, height):
        super().__init__(width, height)
        # Filter parameter - kernel size for low-pass filter (must be odd)
        self.blur_kernel = DEFAULT_BLUR_KERNEL  # Larger kernel = more high-frequency details

    @classmethod
    def get_name(cls):
        return "Frequency Filter"

    @classmethod
    def get_description(cls):
        return "High-pass filter showing fine details and edges"

    @classmethod
    def get_category(cls):
        return "signals"

    def draw(self, frame, face_mask=None):
        """Apply high-pass filter to extract fine details"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create low-pass version by blurring
        if self.blur_kernel > 1:
            low_pass = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        else:
            low_pass = gray

        # High-pass = abs(original - low-pass)
        # Use absolute difference so no signal = 0 (black background)
        high_pass = cv2.absdiff(gray, low_pass)

        # Convert back to 3-channel for display
        result = cv2.cvtColor(high_pass, cv2.COLOR_GRAY2BGR)

        return result

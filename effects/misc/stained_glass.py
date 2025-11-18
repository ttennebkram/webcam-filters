"""
Stained Glass Effect

Creates a vibrant stained glass window appearance using color quantization,
k-means clustering, and edge detection to create distinct colored segments
with dark leading between them.
"""

import cv2
import numpy as np
from core.base_effect import BaseEffect


class StainedGlassEffect(BaseEffect):
    """Stained glass window effect with color quantization and leading"""

    def __init__(self, width, height):
        super().__init__(width, height)

    @classmethod
    def get_name(cls):
        return "Stained Glass"

    @classmethod
    def get_description(cls):
        return "Vibrant stained glass window with color segments and dark leading"

    @classmethod
    def get_category(cls):
        return "misc"

    def draw(self, frame, face_mask=None):
        """Stained glass effect - color quantization and segmentation"""
        # Downsample to 33% for 9x speed improvement
        original_height, original_width = frame.shape[:2]
        small_frame = cv2.resize(frame, (original_width // 3, original_height // 3), interpolation=cv2.INTER_LINEAR)

        # Boost saturation first
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5, beta=0)
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Apply bilateral filter to smooth colors while preserving edges
        smooth = cv2.bilateralFilter(saturated_frame, 9, 75, 75)

        # Color quantization - reduce to limited palette (stained glass colors)
        # Reshape for k-means
        Z = smooth.reshape((-1, 3))
        Z = np.float32(Z)

        # K-means clustering to group similar colors
        K = 24  # Number of color segments (like stained glass pieces)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert back to 8-bit values
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape((small_frame.shape))

        # Apply median filter to create smoother, more uniform "glass pieces"
        result = cv2.medianBlur(result, 5)

        # Optional: Add dark edges between segments for leading effect
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Create black lines for leading
        result[edges > 0] = [0, 0, 0]

        # Upscale back to original size using NEAREST to maintain blocky stained glass look
        result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        return result

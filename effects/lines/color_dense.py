"""
Color Dense Lines Effect

Extracts edges from each RGB channel separately and combines them with color,
creating colorful dense edge visualizations from bit-plane processing.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


class LinesColorDense(BaseEffect):
    """Dense colored edge detection using bit-plane analysis"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    @classmethod
    def get_name(cls) -> str:
        return "Color Dense Lines"

    @classmethod
    def get_description(cls) -> str:
        return "Dense colored edges extracted from RGB bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw the effect - extract colored edges from bit planes"""
        # Convert to HSV to manipulate saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Boost saturation significantly
        hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=2.5, beta=0)  # 2.5x saturation

        # Convert back to BGR
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Dim the saturated frame for background
        background = cv2.convertScaleAbs(saturated_frame, alpha=0.5, beta=0)

        # Create edge-detected overlay with color
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Soften edges with blur
        edges = cv2.GaussianBlur(edges, (5, 5), 0)

        # Create colored edge overlay by masking saturated frame with edges
        edge_mask = cv2.merge([edges, edges, edges])  # 3-channel mask
        edge_overlay = cv2.bitwise_and(saturated_frame, edge_mask)
        edge_overlay = cv2.convertScaleAbs(edge_overlay, alpha=0.45, beta=0)

        # Combine background with colored edge overlay
        background = cv2.addWeighted(background, 1.0, edge_overlay, 1.0, 0)

        # Start with this background
        result = background.copy()

        # Create Canny edge layer for each bit plane of each RGB channel
        # Start with a floating-point accumulator for blending
        edge_accumulator = np.zeros(frame.shape, dtype=np.float32)

        # Split into BGR channels
        b_channel, g_channel, r_channel = cv2.split(saturated_frame)

        # Process each color channel
        channels = [
            (b_channel, np.array([1.0, 0.0, 0.0], dtype=np.float32)),  # Blue
            (g_channel, np.array([0.0, 1.0, 0.0], dtype=np.float32)),  # Green
            (r_channel, np.array([0.0, 0.0, 1.0], dtype=np.float32))   # Red
        ]

        for channel_data, color_mask in channels:
            # Process each bit plane (8 bits, MSB to LSB)
            for bit in range(7, -1, -1):  # 7 is MSB, 0 is LSB
                # Extract bit plane
                bit_plane = ((channel_data >> bit) & 1) * 255
                bit_plane = bit_plane.astype(np.uint8)

                # Apply Canny edge detection to this bit plane
                blurred = cv2.GaussianBlur(bit_plane, (3, 3), 0)
                edges = cv2.Canny(blurred, 50, 150)

                # MSB = thicker, LSB = thinner
                # Thickness: Only MSB planes get slight dilation
                if bit >= 6:  # Only bits 6 and 7 (top 2 MSB)
                    kernel = np.ones((3, 3), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)  # Just 1 iteration

                # Soften edges
                edges = cv2.GaussianBlur(edges, (3, 3), 0)

                # Brightness: Keep intensity more consistent across bit planes
                # The thickness variation does most of the work
                intensity = (bit + 1) / 8.0  # Linear falloff

                # Overall reduction factor
                intensity *= 0.5  # Reduce all edges to 50% intensity

                # Create colored edge layer for this bit plane
                # Convert edges to float and normalize
                edges_float = edges.astype(np.float32) / 255.0

                # Apply intensity scaling
                edges_float *= intensity * 255.0

                # Add to accumulator with color (transparent blending)
                for c in range(3):
                    edge_accumulator[:, :, c] += edges_float * color_mask[c]

        # Normalize and convert accumulator to uint8
        # Clip values to prevent overflow
        edge_accumulator = np.clip(edge_accumulator, 0, 255)
        char_layer = edge_accumulator.astype(np.uint8)

        # Composite edge layer on top of background
        result = cv2.addWeighted(result, 1.0, char_layer, 1.0, 0)

        return result

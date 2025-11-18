"""
Mono 24-Channels Lines Effect

Creates pen and ink style visualization by extracting edges from all 24 bit planes
(8 bits each for grayscale), producing dark lines on white background.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


class LinesMono24Channels(BaseEffect):
    """Pen and ink style - dark lines on white background from bit planes"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    @classmethod
    def get_name(cls) -> str:
        return "Mono 24-Channels"

    @classmethod
    def get_description(cls) -> str:
        return "Pen and ink style using all 8 grayscale bit planes"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw pen and ink style - dark lines on white background"""
        # Convert to grayscale for B&W processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Start with white background
        result = np.ones_like(frame) * 255

        # Create Canny edge layer for each bit plane
        # Start with a floating-point accumulator for B&W blending
        edge_accumulator = np.zeros(gray.shape, dtype=np.float32)

        # Process grayscale channel for all bit planes
        for bit in range(7, -1, -1):  # 7 is MSB, 0 is LSB
            # Extract bit plane from grayscale
            bit_plane = ((gray >> bit) & 1) * 255
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

            # Darkness: Keep intensity more consistent across bit planes
            # The thickness variation does most of the work
            intensity = (bit + 1) / 8.0  # Linear falloff

            # Overall darkness factor
            intensity *= 0.8  # 80% darkness for edges

            # Convert edges to float and normalize
            edges_float = edges.astype(np.float32) / 255.0

            # Apply intensity scaling
            edges_float *= intensity * 255.0

            # Add to accumulator (darker = higher values, will be inverted)
            edge_accumulator += edges_float

        # Normalize and convert accumulator to uint8
        # Clip values to prevent overflow
        edge_accumulator = np.clip(edge_accumulator, 0, 255)
        edges_grayscale = edge_accumulator.astype(np.uint8)

        # Invert: dark lines on white background
        edges_inverted = 255 - edges_grayscale

        # Convert to 3-channel for compositing
        edges_3channel = cv2.merge([edges_inverted, edges_inverted, edges_inverted])

        # Composite: subtract dark lines from white background
        result = cv2.subtract(result, 255 - edges_3channel)

        return result

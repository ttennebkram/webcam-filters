"""
Square Lenses Refraction Effect

Grid of square lenses with fisheye distortion that magnify and refract
the image, creating a multi-faceted view.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


class RefractionSquareLenses(BaseEffect):
    """Square lenses with fisheye effect - grid of magnified regions"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Lens size
        self.lens_size = 80

        # Pre-calculate fisheye mapping once
        self.offset_x, self.offset_y = self.create_fisheye_map(self.lens_size)

        # Calculate grid
        self.cols = width // self.lens_size
        self.rows = height // self.lens_size

    @classmethod
    def get_name(cls) -> str:
        return "Square Lenses"

    @classmethod
    def get_description(cls) -> str:
        return "Grid of square lenses with fisheye magnification effect"

    @classmethod
    def get_category(cls) -> str:
        return "refraction"

    def create_fisheye_map(self, size):
        """Pre-calculate fisheye mapping for a given lens size"""
        half_size = size // 2
        strength = 1.5

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:size, 0:size]

        # Normalized coordinates from -1 to 1
        nx = (x_coords - half_size) / half_size
        ny = (y_coords - half_size) / half_size

        # Distance from center
        r = np.sqrt(nx * nx + ny * ny)

        # Apply fisheye distortion
        theta = np.arctan2(ny, nx)
        r_distorted = r * (1 + strength * r * r)

        # Source coordinates (relative to lens center)
        offset_x = (r_distorted * half_size * np.cos(theta)).astype(np.int32)
        offset_y = (r_distorted * half_size * np.sin(theta)).astype(np.int32)

        return offset_x, offset_y

    def apply_fisheye_to_region(self, frame, center_x, center_y, size, offset_x, offset_y):
        """Apply pre-calculated fisheye distortion to a square region"""
        height, width = frame.shape[:2]

        # Calculate source coordinates
        source_x = center_x + offset_x
        source_y = center_y + offset_y

        # Clip to valid bounds
        source_x = np.clip(source_x, 0, width - 1)
        source_y = np.clip(source_y, 0, height - 1)

        # Use advanced indexing to grab all pixels at once
        lens_region = frame[source_y, source_x]

        return lens_region

    def update(self):
        """Update method - not needed for square lenses"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Square lenses with fisheye effect - grid of magnified regions"""
        # Create result image
        result = np.zeros_like(frame)

        # Draw each lens
        for row in range(self.rows):
            for col in range(self.cols):
                # Position in result image
                result_x = col * self.lens_size
                result_y = row * self.lens_size

                # Center point in source image for this lens
                center_x = result_x + self.lens_size // 2
                center_y = result_y + self.lens_size // 2

                # Apply pre-calculated fisheye to this region
                lens_region = self.apply_fisheye_to_region(frame, center_x, center_y,
                                                          self.lens_size, self.offset_x, self.offset_y)

                # Place lens in result
                result[result_y:result_y+self.lens_size, result_x:result_x+self.lens_size] = lens_region

                # Draw border around lens
                cv2.rectangle(result,
                            (result_x, result_y),
                            (result_x + self.lens_size, result_y + self.lens_size),
                            (50, 50, 50), 1)

        return result

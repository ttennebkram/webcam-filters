"""
Cut Glass Refraction Effect

Static geometric facets that refract the image, creating a cut-glass or
prismatic effect with square-framed facets and beveled edges.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


class RefractionCutGlass(BaseEffect):
    """Cut glass effect - static geometric facets that refract the image"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Create static cut-glass facet pattern
        # Facets are diamond/triangular shapes that create prismatic refraction
        self.facet_size = 120  # Size of each facet

        # Pre-calculate the entire cut-glass mask for the screen
        self.refraction_map = self.create_cut_glass_pattern()

    @classmethod
    def get_name(cls) -> str:
        return "Cut Glass"

    @classmethod
    def get_description(cls) -> str:
        return "Static cut-glass pattern with prismatic refraction effects"

    @classmethod
    def get_category(cls) -> str:
        return "refraction"

    def create_cut_glass_pattern(self):
        """Pre-calculate refraction displacement map for entire screen with square-framed facets"""
        # Create displacement maps for entire screen
        offset_x = np.zeros((self.height, self.width), dtype=np.float32)
        offset_y = np.zeros((self.height, self.width), dtype=np.float32)
        alpha = np.zeros((self.height, self.width), dtype=np.float32)

        facet_size = self.facet_size

        # Create a regular grid of square facets
        for row in range(0, self.height, facet_size):
            for col in range(0, self.width, facet_size):
                # Create square facet at this position
                self.add_square_facet(offset_x, offset_y, alpha, col, row, facet_size)

        # Blur the displacement maps for smoother transitions between facets
        offset_x = cv2.GaussianBlur(offset_x, (5, 5), 0)
        offset_y = cv2.GaussianBlur(offset_y, (5, 5), 0)

        return {
            'offset_x': offset_x,
            'offset_y': offset_y,
            'alpha': alpha
        }

    def add_square_facet(self, offset_x, offset_y, alpha, corner_x, corner_y, size):
        """Add a single square-framed facet with beveled edges and indented center"""
        # Calculate center of the square
        center_x = corner_x + size // 2
        center_y = corner_y + size // 2

        # Frame thickness
        frame_thickness = 0  # No dark frame between squares
        # Flat raised border width (no refraction) - 8 pixels
        flat_border = 8
        # Transition zone from flat to full refraction
        transition_zone = size // 16  # Smaller bevel slope (~7.5 pixels)

        for y in range(max(0, corner_y), min(self.height, corner_y + size)):
            for x in range(max(0, corner_x), min(self.width, corner_x + size)):
                # Distance from square boundaries
                dist_left = x - corner_x
                dist_right = (corner_x + size) - x
                dist_top = y - corner_y
                dist_bottom = (corner_y + size) - y

                # Distance from center
                dx = x - center_x
                dy = y - center_y

                # Minimum distance to any edge
                min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

                if min_edge_dist < frame_thickness:
                    # In the dark frame - no refraction
                    alpha[y, x] = 0.0
                elif min_edge_dist < flat_border:
                    # In the flat raised border area - minimal or no refraction
                    offset_x[y, x] = 0.0
                    offset_y[y, x] = 0.0
                    alpha[y, x] = 0.0  # Flat, no distortion
                elif min_edge_dist < flat_border + transition_zone:
                    # In bevel zone - refraction goes from center toward the raised edge
                    # This is the slope from valley (center) up to raised flat border
                    transition_progress = (min_edge_dist - flat_border) / transition_zone

                    # Inverse progress: 1.0 at inner edge (near valley), 0.0 at outer edge (near flat)
                    bevel_strength = 1.0 - transition_progress

                    # Refraction pushes TOWARD the nearest edge (upward slope effect)
                    # Determine which edge is closest
                    if dist_left == min_edge_dist:
                        # Left edge is closest - push toward left
                        refract_x = -35.0 * bevel_strength
                        refract_y = 0.0
                    elif dist_right == min_edge_dist:
                        # Right edge is closest - push toward right
                        refract_x = 35.0 * bevel_strength
                        refract_y = 0.0
                    elif dist_top == min_edge_dist:
                        # Top edge is closest - push toward top
                        refract_x = 0.0
                        refract_y = -35.0 * bevel_strength
                    else:
                        # Bottom edge is closest - push toward bottom
                        refract_x = 0.0
                        refract_y = 35.0 * bevel_strength

                    offset_x[y, x] = refract_x
                    offset_y[y, x] = refract_y
                    alpha[y, x] = 1.0 * bevel_strength
                else:
                    # In the indented valley center - create crosshatch diagonal pattern
                    # Diagonal lines run at 45 degrees in both directions
                    diagonal_spacing = 40  # Space between diagonal ridges
                    diagonal_width = 1  # Width of each diagonal ridge

                    # Calculate position along both diagonals (45 degree angles)
                    diagonal_pos1 = (dx + dy) % diagonal_spacing  # One direction
                    diagonal_pos2 = (dx - dy) % diagonal_spacing  # Other direction

                    # Distance from center of diagonal ridges in both directions
                    dist_from_ridge1 = abs(diagonal_pos1 - diagonal_spacing / 2)
                    dist_from_ridge2 = abs(diagonal_pos2 - diagonal_spacing / 2)

                    # Use the minimum distance (closer to either diagonal)
                    dist_from_ridge = min(dist_from_ridge1, dist_from_ridge2)

                    if dist_from_ridge < diagonal_width:
                        # On a diagonal ridge - more pronounced elevation
                        ridge_roundness = 1.0 - (dist_from_ridge / diagonal_width)
                        ridge_height = ridge_roundness * 0.6

                        # Very minimal refraction on ridge tops for sharper effect
                        refract_x = 3.0 * (1.0 - ridge_height)
                        refract_y = 3.0 * (1.0 - ridge_height)
                        alpha[y, x] = 0.2 + 0.5 * ridge_height
                    else:
                        # Between ridges - valley bottom with strong refraction
                        valley_depth = (dist_from_ridge - diagonal_width) / (diagonal_spacing / 2 - diagonal_width)
                        valley_depth = min(1.0, valley_depth)

                        # Strong refraction in valleys for dramatic visibility
                        if abs(dx) > abs(dy):
                            if dx > 0:
                                refract_x = 70.0 * valley_depth
                                refract_y = (dy / abs(dx) * 45.0 * valley_depth) if abs(dx) > 0 else 0.0
                            else:
                                refract_x = -70.0 * valley_depth
                                refract_y = (dy / abs(dx) * 45.0 * valley_depth) if abs(dx) > 0 else 0.0
                        else:
                            if dy > 0:
                                refract_y = 70.0 * valley_depth
                                refract_x = (dx / abs(dy) * 45.0 * valley_depth) if abs(dy) > 0 else 0.0
                            else:
                                refract_y = -70.0 * valley_depth
                                refract_x = (dx / abs(dy) * 45.0 * valley_depth) if abs(dy) > 0 else 0.0

                        alpha[y, x] = 1.0 * valley_depth

                    offset_x[y, x] = refract_x
                    offset_y[y, x] = refract_y

    def update(self):
        """Update animation - static effect, no updates needed"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply cut-glass facet refraction to the entire frame"""
        # Use original frame without enhancement
        enhanced_frame = frame.copy()

        # Get pre-calculated refraction maps
        offset_x = self.refraction_map['offset_x']
        offset_y = self.refraction_map['offset_y']
        alpha_map = self.refraction_map['alpha']

        # Create coordinate grids for entire frame
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]

        # Calculate source coordinates with refraction
        source_x = (x_coords + offset_x.astype(np.int32)).clip(0, self.width - 1)
        source_y = (y_coords + offset_y.astype(np.int32)).clip(0, self.height - 1)

        # Get refracted pixels from enhanced frame
        refracted = enhanced_frame[source_y, source_x]

        # Alpha blend with original frame
        alpha_3d = alpha_map[:, :, np.newaxis]  # Add channel dimension
        result = (refracted * alpha_3d + frame * (1.0 - alpha_3d)).astype(np.uint8)

        return result

"""
Simple passthrough effect for testing.

This effect does nothing - it just returns the frame unchanged.
"""

import numpy as np
from core.base_effect import BaseEffect


class PassthroughEffect(BaseEffect):
    """A simple effect that passes frames through unchanged"""

    @classmethod
    def get_name(cls) -> str:
        return "Passthrough (No Effect)"

    @classmethod
    def get_description(cls) -> str:
        return "Returns the original frame without any modifications"

    @classmethod
    def get_category(cls) -> str:
        return "misc"

    def draw(self, frame: np.ndarray, face_mask=None) -> np.ndarray:
        """Return the frame unchanged"""
        return frame

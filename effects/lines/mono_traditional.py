"""
Mono Traditional Canny Lines Effect

Traditional Canny edge detection with temporal smoothing and high-pass texture
for detailed, stable pen-and-ink style visualization.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


class LinesMonoTraditional(BaseEffect):
    """Traditional Canny edge detection - dark lines on white background with texture"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Temporal smoothing buffer to reduce flicker
        self.prev_frame = None
        self.prev_gray = None
        self.alpha = 0.3  # Blend factor: 0.3 new frame + 0.7 previous frame (heavy smoothing)
        self.target_mean = 128  # Target mean brightness
        self.current_mean = 128  # Smoothed mean brightness

    @classmethod
    def get_name(cls) -> str:
        return "Mono Traditional Canny"

    @classmethod
    def get_description(cls) -> str:
        return "Traditional Canny with temporal smoothing and high-pass texture"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

    def update(self):
        """Update - not needed for static effect"""
        pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Traditional Canny edge detection - dark lines on white background with texture"""
        # Temporal smoothing to reduce flicker from camera auto-adjustments
        if self.prev_frame is not None:
            frame = cv2.addWeighted(frame, self.alpha, self.prev_frame, 1 - self.alpha, 0)
        self.prev_frame = frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize brightness gradually to combat auto-exposure flickering
        # Smooth the mean brightness over time
        frame_mean = gray.mean()
        self.current_mean = self.current_mean * 0.9 + frame_mean * 0.1  # Slowly adapt

        # Adjust to target brightness without dramatic jumps
        if self.current_mean > 0:
            adjustment = self.target_mean / self.current_mean
            # Clamp adjustment to prevent dramatic changes
            adjustment = np.clip(adjustment, 0.8, 1.2)
            gray = cv2.convertScaleAbs(gray, alpha=adjustment, beta=0)

        # Also smooth the grayscale temporally for even more stability
        if self.prev_gray is not None:
            gray = cv2.addWeighted(gray, self.alpha, self.prev_gray, 1 - self.alpha, 0)
        self.prev_gray = gray.copy()

        # Create high-pass filter for texture (use VERY small blur to detect fine detail)
        low_pass = cv2.GaussianBlur(gray, (3, 3), 0)  # Even smaller blur = even finer detail
        high_pass = cv2.subtract(gray.astype(np.int16), low_pass.astype(np.int16))

        # Expand contrast MASSIVELY - multiply the range
        high_pass = high_pass.astype(np.float32) * 25.0  # Amplify detail 25x!!!

        # Normalize to use full range (expand contrast to maximum)
        high_pass_min = high_pass.min()
        high_pass_max = high_pass.max()
        if high_pass_max - high_pass_min > 0:
            # Stretch to full 0-255 range
            high_pass = ((high_pass - high_pass_min) / (high_pass_max - high_pass_min) * 510) - 255  # Double range, centered at 0
        else:
            high_pass = np.zeros_like(high_pass)

        # Desaturate the original frame
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        desaturated_bg = cv2.convertScaleAbs(gray_3channel, alpha=0.15, beta=230)  # Very light background

        # Start with desaturated background
        result = desaturated_bg.copy()

        # Apply multiple Canny detections at different sensitivities
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Strong edges (lower thresholds to catch more detail)
        edges_strong = cv2.Canny(blurred, 30, 90)

        # Medium edges (lower thresholds for better retention)
        edges_medium = cv2.Canny(blurred, 15, 50)

        # Faint edges (very low thresholds for subtle detail)
        edges_faint = cv2.Canny(blurred, 8, 25)

        # Extra faint edges (even more sensitive to hold onto subtle lines)
        edges_extra_faint = cv2.Canny(blurred, 4, 15)

        # Very extra faint edges (maximum sensitivity)
        edges_very_extra_faint = cv2.Canny(blurred, 2, 10)

        # Blur each layer differently for varied softness
        edges_strong = cv2.GaussianBlur(edges_strong, (9, 9), 0)  # Much more blur for smooth main lines
        edges_medium = cv2.GaussianBlur(edges_medium, (5, 5), 0)  # More blur
        edges_faint = cv2.GaussianBlur(edges_faint, (3, 3), 0)    # Light blur
        edges_extra_faint = cv2.GaussianBlur(edges_extra_faint, (3, 3), 0)  # Light blur
        edges_very_extra_faint = cv2.GaussianBlur(edges_very_extra_faint, (3, 3), 0)  # Light blur

        # Combine with different intensities - boost faint lines significantly
        combined = np.zeros_like(gray, dtype=np.float32)
        combined += edges_strong.astype(np.float32) * 0.7   # Strong lines at 70%
        combined += edges_medium.astype(np.float32) * 0.7   # Medium lines at 70% (boosted)
        combined += edges_faint.astype(np.float32) * 0.65   # Faint lines at 65% (boosted!)
        combined += edges_extra_faint.astype(np.float32) * 0.6  # Extra faint lines at 60% (boosted!)
        combined += edges_very_extra_faint.astype(np.float32) * 0.5  # Very extra faint at 50%

        # Add high-pass texture with MASSIVE impact
        high_pass_contribution = high_pass * 8.0  # EXTREME contribution of texture!!!
        combined += high_pass_contribution

        # Normalize and clip
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Invert: we want dark lines on white background
        edges_inverted = 255 - combined

        # Convert to 3-channel
        edges_3channel = cv2.merge([edges_inverted, edges_inverted, edges_inverted])

        # Composite onto background (darken where edges exist)
        result = cv2.multiply(result.astype(np.float32) / 255.0,
                             edges_3channel.astype(np.float32) / 255.0)
        result = (result * 255).astype(np.uint8)

        return result

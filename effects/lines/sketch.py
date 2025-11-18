"""
Sketch Lines Effect

Advanced sketch effect combining Canny edge detection with FFT-based high-pass
filtering for detailed pen-and-ink style visualization. This is a simplified
version without the UI control panel.
"""

import cv2
import numpy as np
from typing import Optional
from core.base_effect import BaseEffect


# Configuration constants
DEFAULT_CANNY_BLUR_KERNEL = 3
DEFAULT_THRESHOLD1 = 25
DEFAULT_THRESHOLD2 = 7
DEFAULT_APERTURE_SIZE = 3
DEFAULT_L2_GRADIENT = True
DEFAULT_FFT_RADIUS = 30
DEFAULT_FFT_SMOOTHNESS = 0
DEFAULT_FFT_BOOST = 2.5
DEFAULT_APPLY_CANNY = True
DEFAULT_APPLY_FFT = True
DEFAULT_INVERT = True


class LinesSketch(BaseEffect):
    """Advanced sketch combining Canny and FFT filtering"""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

        # Canny parameters
        self.canny_blur = DEFAULT_CANNY_BLUR_KERNEL
        self.threshold1 = DEFAULT_THRESHOLD1
        self.threshold2 = DEFAULT_THRESHOLD2
        self.aperture_size = DEFAULT_APERTURE_SIZE
        self.l2_gradient = DEFAULT_L2_GRADIENT

        # FFT parameters
        self.fft_radius = DEFAULT_FFT_RADIUS
        self.fft_smoothness = DEFAULT_FFT_SMOOTHNESS
        self.fft_boost = DEFAULT_FFT_BOOST

        # Control flags
        self.apply_canny = DEFAULT_APPLY_CANNY
        self.apply_fft = DEFAULT_APPLY_FFT
        self.invert = DEFAULT_INVERT

    @classmethod
    def get_name(cls) -> str:
        return "Sketch (Canny + FFT)"

    @classmethod
    def get_description(cls) -> str:
        return "Advanced sketch combining Canny edge detection and FFT filtering"

    @classmethod
    def get_category(cls) -> str:
        return "lines"

    def update(self):
        """Update - not needed for static effect"""
        pass

    def apply_canny_edges(self, frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.canny_blur > 1:
            blurred = cv2.GaussianBlur(gray, (self.canny_blur, self.canny_blur), 0)
        else:
            blurred = gray

        edges = cv2.Canny(blurred, self.threshold1, self.threshold2,
                         apertureSize=self.aperture_size, L2gradient=self.l2_gradient)

        return edges

    def apply_fft_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply FFT-based high-pass filter"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute FFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get dimensions
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create high-pass mask (reject low frequencies in center)
        center_y, center_x = crow, ccol
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        if self.fft_smoothness == 0:
            # Hard circle mask
            mask = np.ones((rows, cols, 2), np.float32)
            mask_area = distance <= self.fft_radius
            mask[mask_area] = 0
        else:
            # Smooth transition
            sigma = self.fft_smoothness / 10.0
            mask = np.ones((rows, cols, 2), np.float32)
            transition = 1.0 / (1.0 + np.exp(-(distance - self.fft_radius) / sigma))
            mask[:, :, 0] = transition
            mask[:, :, 1] = transition

        # Apply mask to FFT
        fshift = dft_shift * mask

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        high_pass = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply boost
        if self.fft_boost != 0:
            high_pass_float = high_pass.astype(np.float32)
            high_pass_float = high_pass_float * (1.0 + self.fft_boost)
            high_pass = np.clip(high_pass_float, 0, 255).astype(np.uint8)

        return high_pass

    def draw(self, frame: np.ndarray, face_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply combined filters and return result"""
        result = np.zeros((self.height, self.width), dtype=np.uint8)

        # Apply Canny if enabled
        if self.apply_canny:
            canny_result = self.apply_canny_edges(frame)
            result = cv2.add(result, canny_result)

        # Apply FFT filter if enabled
        if self.apply_fft:
            fft_result = self.apply_fft_filter(frame)
            result = cv2.add(result, fft_result)

        # Invert if enabled
        if self.invert:
            result = cv2.bitwise_not(result)

        # Convert to 3-channel for display
        result_3channel = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result_3channel

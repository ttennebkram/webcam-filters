"""
Camera discovery and management utilities.
"""

import cv2
import platform


def find_cameras(max_cameras=10):
    """Find all available cameras

    Args:
        max_cameras: Maximum number of cameras to check

    Returns:
        List of camera indices that are available
    """
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return available_cameras


def get_camera_name(index):
    """Get a friendly name for a camera

    Args:
        index: Camera index

    Returns:
        String description of the camera
    """
    # On macOS, camera 0 is usually built-in
    if platform.system() == 'Darwin' and index == 0:
        return "Built-in Camera"
    else:
        return f"Camera {index}"


def open_camera(index, width=None, height=None, warmup_frames=5):
    """Open a camera with optional resolution

    Args:
        index: Camera index
        width: Desired width (or None for default)
        height: Desired height (or None for default)
        warmup_frames: Number of frames to discard for camera warmup (default: 5)

    Returns:
        cv2.VideoCapture object or None if failed
    """
    import time

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Give camera time to initialize
    time.sleep(0.5)

    # Discard initial frames to allow camera to adjust exposure/white balance
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.1)

    return cap

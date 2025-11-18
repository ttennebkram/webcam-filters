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
    return f"Camera {index}"


def open_camera(index, width=None, height=None):
    """Open a camera with optional resolution

    Args:
        index: Camera index
        width: Desired width (or None for default)
        height: Desired height (or None for default)

    Returns:
        cv2.VideoCapture object or None if failed
    """
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        return None

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap

"""
Blobs pipeline effect.

Stacks blur, grayscale, and threshold effects to create blob detection.
"""

from effects.opencv.pipeline import BasePipelineEffect


class BlobsEffect(BasePipelineEffect):
    """Pipeline for blob detection using blur, grayscale, and threshold"""

    # Define the effects to stack in order
    PIPELINE_EFFECTS = [
        'grayscale',
        'blur',
        'threshold_adaptive',
    ]

    @classmethod
    def get_config_filename(cls) -> str:
        return "blobs"

    @classmethod
    def get_name(cls) -> str:
        return "Blobs"

    @classmethod
    def get_description(cls) -> str:
        return "Blob detection pipeline: blur → grayscale → threshold"

    @classmethod
    def get_method_signature(cls) -> str:
        return "cv2.SimpleBlobDetector_create(params)"

    @classmethod
    def get_category(cls) -> str:
        return "opencv"

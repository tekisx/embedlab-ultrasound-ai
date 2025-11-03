"""
Core processing modules for the Image Embedding Lab.

Contains image processing, embedding computation, and similarity algorithms.
"""

from .image_processor import ImageProcessor, validate_image_quality

__all__ = ['ImageProcessor', 'validate_image_quality']
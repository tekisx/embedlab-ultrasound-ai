"""
Utility modules for the Image Embedding Lab.

Contains PHI-safe logging and configuration management.
"""

from .logging import get_logger, phi_safe_identifier

__all__ = ['get_logger', 'phi_safe_identifier']
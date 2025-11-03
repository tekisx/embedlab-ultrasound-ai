"""
Medical image preprocessing with quality checks and PHI-safe logging.

This module handles all image loading and preprocessing operations,
ensuring quality control for medical imaging applications.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from PIL import Image
import hashlib
from dataclasses import dataclass
import time

from ..utils.logging import get_logger, phi_safe_identifier, log_image_processing
from ..utils.config import Config

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for processed medical images."""

    file_hash: str
    original_path: str  # Will be hashed in logs
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    quality_score: float
    is_valid: bool
    processing_time_ms: float
    warnings: List[str] = None
    phi_safe_id: str = None  # PHI-safe identifier

    def __post_init__(self):
        """Initialize warnings list if not provided and set PHI-safe ID."""
        if self.warnings is None:
            self.warnings = []
        # Ensure PHI-safe ID is set
        if self.phi_safe_id is None:
            self.phi_safe_id = self.file_hash

    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with PHI-safe values."""
        return {
            'file_hash': self.file_hash,
            'phi_safe_id': self.phi_safe_id,
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'format': self.format,
            'size_kb': round(self.size_bytes / 1024, 2),
            'quality_score': round(self.quality_score, 3),
            'is_valid': self.is_valid,
            'processing_time_ms': round(self.processing_time_ms, 2),
            'warning_count': len(self.warnings)
        }


class ImageProcessor:
    """
    Handles medical image loading and preprocessing with quality control.

    Designed for ultrasound and medical imaging with PHI safety.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the image processor.

        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or Config()

        # Image normalization parameters (ImageNet standards)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        logger.info(f"ImageProcessor initialized with image_size={self.config.image_size}")

    def load_image(self,
                  image_path: Union[str, Path],
                  convert_to_rgb: bool = True) -> Tuple[Optional[Image.Image], ImageMetadata]:
        """
        Load an image from disk with quality validation.

        Args:
            image_path: Path to the image file
            convert_to_rgb: Whether to convert grayscale to RGB

        Returns:
            Tuple of (PIL Image or None if invalid, ImageMetadata)
        """
        start_time = time.time()
        image_path = Path(image_path)
        file_hash = phi_safe_identifier(image_path)

        metadata = ImageMetadata(
            file_hash=file_hash,
            original_path=str(image_path),
            width=0,
            height=0,
            channels=0,
            format="unknown",
            size_bytes=0,
            quality_score=0.0,
            is_valid=False,
            processing_time_ms=0.0
        )

        try:
            # Check file exists and size
            if not image_path.exists():
                metadata.warnings.append("File does not exist")
                logger.error(f"File not found: {file_hash}")
                return None, metadata

            metadata.size_bytes = image_path.stat().st_size

            # Check file size limits (medical images shouldn't be too small or too large)
            if metadata.size_bytes < 1024:  # < 1KB
                metadata.warnings.append("File too small (< 1KB)")
                logger.warning(f"File too small: {file_hash}")
                return None, metadata

            if metadata.size_bytes > 100 * 1024 * 1024:  # > 100MB
                metadata.warnings.append("File too large (> 100MB)")
                logger.warning(f"File too large: {file_hash}")
                return None, metadata

            # Load image
            img = Image.open(image_path)
            metadata.format = img.format or "unknown"
            metadata.width = img.width
            metadata.height = img.height

            # Check dimensions
            if (img.width < self.config.min_image_size or
                img.height < self.config.min_image_size):
                metadata.warnings.append(f"Image too small: {img.width}x{img.height}")
                logger.warning(f"Image dimensions too small: {file_hash}")

            if (img.width > self.config.max_image_size or
                img.height > self.config.max_image_size):
                metadata.warnings.append(f"Image too large: {img.width}x{img.height}")
                logger.warning(f"Image dimensions too large: {file_hash}")

            # Handle different image modes
            if img.mode == 'L':  # Grayscale
                metadata.channels = 1
                if convert_to_rgb:
                    img = img.convert('RGB')
                    metadata.channels = 3
                    logger.debug(f"Converted grayscale to RGB: {file_hash}")
            elif img.mode == 'RGB':
                metadata.channels = 3
            elif img.mode == 'RGBA':
                metadata.channels = 4
                img = img.convert('RGB')
                metadata.channels = 3
                logger.debug(f"Converted RGBA to RGB: {file_hash}")
            elif img.mode in ['P', 'PA']:  # Palette mode
                img = img.convert('RGB')
                metadata.channels = 3
                logger.debug(f"Converted palette mode to RGB: {file_hash}")
            else:
                metadata.warnings.append(f"Unsupported image mode: {img.mode}")
                logger.error(f"Unsupported image mode {img.mode}: {file_hash}")
                return None, metadata

            # Quality checks if enabled
            if self.config.check_image_quality:
                quality_score, quality_warnings = self._check_image_quality(img)
                metadata.quality_score = quality_score
                metadata.warnings.extend(quality_warnings)

                if quality_score < 0.3:  # Very low quality
                    metadata.warnings.append(f"Quality score too low: {quality_score:.2f}")
                    if self.config.reject_blank_images:
                        logger.warning(f"Rejecting low quality image: {file_hash}")
                        return None, metadata
            else:
                # Default quality score if checks disabled
                metadata.quality_score = 0.8

            metadata.is_valid = True
            metadata.processing_time_ms = (time.time() - start_time) * 1000

            log_image_processing(logger, image_path, "loaded", metadata.to_safe_dict())

            return img, metadata

        except Exception as e:
            metadata.warnings.append(f"Load error: {str(e)}")
            metadata.processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed to load image {file_hash}: {e}")
            return None, metadata

    def preprocess(self,
                  img: Image.Image,
                  target_size: Optional[int] = None) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            img: PIL Image
            target_size: Target size for resizing. If None, uses config.

        Returns:
            Preprocessed numpy array (C, H, W) normalized for model input
        """
        target_size = target_size or self.config.image_size

        # Resize with high-quality resampling
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {target_size}x{target_size}")

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0  # Scale to [0, 1]

        # Ensure 3 channels
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)

        # ImageNet normalization
        img_array = (img_array - self.mean) / self.std

        # Convert to CHW format (channels first) for PyTorch
        img_array = img_array.transpose(2, 0, 1)

        return img_array.astype(np.float32)

    def preprocess_for_embedding(self, img: Image.Image) -> np.ndarray:
        """
        Preprocess image specifically for embedding generation.

        Args:
            img: PIL Image

        Returns:
            Preprocessed numpy array (1, C, H, W) ready for model input
        """
        preprocessed = self.preprocess(img, target_size=224)  # ResNet50 expects 224x224
        # Add batch dimension
        return np.expand_dims(preprocessed, axis=0)

    def load_batch(self,
                  image_paths: List[Union[str, Path]],
                  convert_to_rgb: bool = True) -> Tuple[List[Optional[Image.Image]], List[ImageMetadata]]:
        """
        Load a batch of images with metadata.

        Args:
            image_paths: List of image paths
            convert_to_rgb: Whether to convert grayscale to RGB

        Returns:
            Tuple of (list of PIL Images or None, list of metadata)
        """
        images = []
        metadatas = []

        for path in image_paths:
            img, metadata = self.load_image(path, convert_to_rgb)
            images.append(img)
            metadatas.append(metadata)

        return images, metadatas

    def _check_image_quality(self, img: Image.Image) -> Tuple[float, List[str]]:
        """
        Check image quality for medical ultrasound imaging standards.

        Args:
            img: PIL Image

        Returns:
            Tuple of (quality_score 0-1, list of warnings)
        """
        warnings = []
        quality_factors = []  # Track individual quality components

        # Convert to numpy for analysis
        img_array = np.array(img)

        # Normalize to 0-1 if needed
        if img_array.max() > 1:
            img_array_norm = img_array / 255.0
        else:
            img_array_norm = img_array

        # 1. Check if image is mostly blank/black
        if self._is_blank_image(img_array):
            warnings.append("Image appears to be nearly blank")
            quality_factors.append(0.1)
        else:
            quality_factors.append(1.0)

        # 2. Check entropy (information content) - ultrasound typically 4-7
        entropy = self._calculate_entropy(img_array)
        if entropy < 2.0:
            warnings.append("Low entropy detected")
            quality_factors.append(0.3)
        elif entropy < 4.0:
            warnings.append("Below average entropy")
            quality_factors.append(0.6)
        elif entropy > 7.5:
            # Very high entropy might indicate noise
            warnings.append("High entropy - possible noise")
            quality_factors.append(0.8)
        else:
            quality_factors.append(1.0)

        # 3. Check contrast - ultrasound images typically have moderate contrast
        contrast = self._calculate_contrast(img_array)
        if contrast < 0.05:
            warnings.append("Low contrast detected")
            quality_factors.append(0.4)
        elif contrast < 0.15:
            quality_factors.append(0.7)
        elif contrast > 0.5:
            # Very high contrast might be abnormal for ultrasound
            quality_factors.append(0.9)
        else:
            quality_factors.append(1.0)

        # 4. Check for extreme brightness
        mean_brightness = img_array_norm.mean()
        if mean_brightness < 0.1:
            warnings.append("Image appears to be very dark")
            quality_factors.append(0.5)
        elif mean_brightness > 0.9:
            warnings.append("Image appears to be very bright")
            quality_factors.append(0.5)
        elif mean_brightness < 0.2 or mean_brightness > 0.8:
            quality_factors.append(0.8)
        else:
            quality_factors.append(1.0)

        # 5. Check for salt-and-pepper noise (common in ultrasound)
        std_dev = img_array_norm.std()
        if std_dev < 0.05:
            warnings.append("Very low variance - possible uniform image")
            quality_factors.append(0.3)
        elif std_dev > 0.4:
            warnings.append("High variance - possible excessive noise")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)

        # Calculate overall quality score (geometric mean for better sensitivity)
        if quality_factors:
            # Use geometric mean to be more sensitive to low scores
            quality_score = np.power(np.prod(quality_factors), 1.0/len(quality_factors))
        else:
            quality_score = 0.5  # Default if no factors calculated

        # Ensure score is in valid range
        quality_score = max(0.1, min(1.0, quality_score))

        return quality_score, warnings

    def _is_blank_image(self, img_array: np.ndarray) -> bool:
        """
        Check if image is blank (all black or all white).

        Args:
            img_array: Image as numpy array

        Returns:
            True if image appears blank
        """
        # Normalize to 0-1 range if needed
        if img_array.max() > 1:
            img_array = img_array / 255.0

        # Check if all pixels are very similar
        std_dev = img_array.std()

        # Check if mostly black or white
        mean_val = img_array.mean()
        is_mostly_black = mean_val < self.config.blank_threshold
        is_mostly_white = mean_val > (1 - self.config.blank_threshold)

        # Low standard deviation indicates uniform image
        is_uniform = std_dev < self.config.blank_threshold

        return is_uniform or (is_mostly_black and std_dev < 0.1) or (is_mostly_white and std_dev < 0.1)

    def _calculate_entropy(self, img_array: np.ndarray) -> float:
        """
        Calculate image entropy (information content).

        Higher entropy indicates more information/detail in the image.

        Args:
            img_array: Image as numpy array

        Returns:
            Entropy value (typically 0-8)
        """
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))

        # Normalize histogram
        hist = hist / hist.sum()

        # Calculate entropy
        # Add small epsilon to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation.

        Args:
            img_array: Image as numpy array

        Returns:
            Contrast value (0-1)
        """
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Normalize to 0-1 if needed
        if gray.max() > 1:
            gray = gray / 255.0

        # Standard deviation as contrast measure
        contrast = gray.std()

        return contrast

    def batch_preprocess(self,
                        image_paths: List[Union[str, Path]],
                        convert_to_rgb: bool = True) -> Tuple[np.ndarray, List[ImageMetadata]]:
        """
        Batch preprocess multiple images.

        Args:
            image_paths: List of image paths
            convert_to_rgb: Whether to convert grayscale to RGB

        Returns:
            Tuple of (batch array (N, C, H, W), list of metadata)
        """
        batch_images = []
        batch_metadata = []

        for path in image_paths:
            img, metadata = self.load_image(path, convert_to_rgb)

            if img is not None:
                preprocessed = self.preprocess(img)
                batch_images.append(preprocessed)
                batch_metadata.append(metadata)
            else:
                logger.warning(f"Skipping invalid image: {phi_safe_identifier(path)}")

        if batch_images:
            batch_array = np.stack(batch_images, axis=0)
            logger.info(f"Batch preprocessed {len(batch_images)} images")
            return batch_array, batch_metadata
        else:
            logger.error("No valid images in batch")
            return np.array([]), batch_metadata


def validate_image_quality(image_path: Union[str, Path],
                          config: Optional[Config] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Standalone function to validate image quality.

    Args:
        image_path: Path to image
        config: Optional configuration

    Returns:
        Tuple of (is_valid, quality_info)
    """
    processor = ImageProcessor(config)
    img, metadata = processor.load_image(image_path)

    quality_info = metadata.to_safe_dict()
    quality_info['warnings'] = metadata.warnings

    return metadata.is_valid, quality_info


if __name__ == "__main__":
    # Example usage
    from ..utils.config import get_default_config

    config = get_default_config()
    processor = ImageProcessor(config)

    # Test with a sample image
    test_image = Path("test_ultrasound.jpg")
    if test_image.exists():
        img, metadata = processor.load_image(test_image)
        if img:
            preprocessed = processor.preprocess(img)
            print(f"Preprocessed shape: {preprocessed.shape}")
            print(f"Metadata: {metadata.to_safe_dict()}")
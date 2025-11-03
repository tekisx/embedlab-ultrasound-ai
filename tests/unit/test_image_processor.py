"""
Unit tests for image processing module using REAL ultrasound images.

Tests medical image processing capabilities with actual ultrasound data.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
from PIL import Image
import hashlib
from typing import Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.image_processor import ImageProcessor, ImageMetadata
from src.utils.config import Config


class TestImageProcessor:
    """Test suite for ImageProcessor using REAL ultrasound images."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance with test config."""
        config = Config()
        return ImageProcessor(config)

    @pytest.fixture
    def real_ultrasound_path(self):
        """Get path to a real ultrasound image from our dataset."""
        # Use actual breast ultrasound images we downloaded
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT")

        # Try to find a benign image first
        benign_path = breast_path / "benign"
        if benign_path.exists():
            images = list(benign_path.glob("*.png"))
            if images and images[0].exists():
                return images[0]

        # Fallback to any available image
        for img_path in breast_path.rglob("*.png"):
            if "mask" not in img_path.name.lower():
                return img_path

        # If no real images found, skip test
        pytest.skip("No real ultrasound images found. Run download_real_datasets.py first.")

    def test_load_real_ultrasound_image(self, processor, real_ultrasound_path):
        """Test loading an actual ultrasound image from Kaggle dataset."""
        img, metadata = processor.load_image(real_ultrasound_path)

        assert img is not None
        assert metadata.is_valid
        assert metadata.width > 0
        assert metadata.height > 0
        assert metadata.format in ["PNG", "JPEG", "BMP", "TIFF"]
        assert 0.0 <= metadata.quality_score <= 1.0
        assert metadata.phi_safe_id is not None
        assert len(metadata.phi_safe_id) == 16  # SHA256 truncated

    def test_preprocess_medical_image(self, processor, real_ultrasound_path):
        """Test preprocessing pipeline on real medical ultrasound."""
        img, _ = processor.load_image(real_ultrasound_path)

        if img is None:
            pytest.skip("Could not load image")

        processed = processor.preprocess_for_embedding(img)

        # Check output shape for ResNet50
        assert processed.shape == (1, 3, 224, 224)
        assert processed.dtype == np.float32

        # Check normalization (ImageNet stats)
        assert -3.0 <= processed.min() <= 3.0  # Roughly ±3 std devs
        assert -3.0 <= processed.max() <= 3.0

    def test_quality_assessment_on_real_data(self, processor):
        """Test quality metrics on various real ultrasound images."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT")

        quality_scores = []
        categories = {"benign": [], "malignant": [], "normal": []}

        for category in categories.keys():
            cat_path = breast_path / category
            if cat_path.exists():
                # Sample a few images from each category
                images = list(cat_path.glob("*.png"))[:5]

                for img_path in images:
                    if "mask" not in img_path.name.lower():
                        _, metadata = processor.load_image(img_path)
                        if metadata.is_valid:
                            categories[category].append(metadata.quality_score)
                            quality_scores.append(metadata.quality_score)

        if quality_scores:
            # Real ultrasound images should have reasonable quality
            avg_quality = sum(quality_scores) / len(quality_scores)
            assert 0.2 <= avg_quality <= 1.0, f"Unexpected average quality: {avg_quality}"
            # Most medical ultrasound images should have good quality
            assert avg_quality >= 0.5, f"Quality too low for medical images: {avg_quality}"

            # Different categories might have different quality distributions
            for cat, scores in categories.items():
                if scores:
                    print(f"{cat.capitalize()} avg quality: {sum(scores)/len(scores):.3f}")

    def test_edge_case_detection(self, processor):
        """Test detection of problematic ultrasound images."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT")

        edge_cases = {
            "low_quality": [],
            "very_dark": [],
            "small_images": []
        }

        # Check a sample of images for edge cases
        for img_path in list(breast_path.rglob("*.png"))[:20]:
            if "mask" in img_path.name.lower():
                continue

            img, metadata = processor.load_image(img_path)

            if metadata.quality_score < 0.3:
                edge_cases["low_quality"].append(img_path.name)

            if img is not None:
                img_array = np.array(img)
                if img_array.mean() < 50:
                    edge_cases["very_dark"].append(img_path.name)

            if metadata.width < 200 or metadata.height < 200:
                edge_cases["small_images"].append(img_path.name)

        # Report findings (these are expected in medical datasets)
        for case_type, cases in edge_cases.items():
            if cases:
                print(f"\nFound {len(cases)} {case_type} images (expected in medical data)")

    def test_batch_processing(self, processor):
        """Test batch processing of multiple ultrasound images."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign")

        if not breast_path.exists():
            pytest.skip("Benign ultrasound folder not found")

        # Get first 5 non-mask images
        batch_paths = []
        for img_path in breast_path.glob("*.png"):
            if "mask" not in img_path.name.lower():
                batch_paths.append(img_path)
                if len(batch_paths) == 5:
                    break

        if len(batch_paths) < 2:
            pytest.skip("Not enough images for batch test")

        images, metadatas = processor.load_batch(batch_paths)

        assert len(images) == len(batch_paths)
        assert len(metadatas) == len(batch_paths)

        # Check all loaded successfully
        valid_count = sum(1 for m in metadatas if m.is_valid)
        assert valid_count > 0, "No valid images in batch"

    def test_grayscale_ultrasound_handling(self, processor, real_ultrasound_path):
        """Test handling of grayscale ultrasound images (common in medical imaging)."""
        img, metadata = processor.load_image(real_ultrasound_path)

        if img is None:
            pytest.skip("Could not load image")

        # Many ultrasound images are grayscale
        if metadata.channels == 1:
            # Should still be able to preprocess for RGB model
            processed = processor.preprocess_for_embedding(img)
            assert processed.shape == (1, 3, 224, 224)

            # Check that grayscale was properly converted to RGB
            # (all channels should be similar)
            channel_diff = np.abs(processed[0, 0] - processed[0, 1]).mean()
            assert channel_diff < 0.01, "Grayscale to RGB conversion issue"

    def test_phi_safety(self, processor, real_ultrasound_path):
        """Test PHI-safe identifier generation."""
        _, metadata1 = processor.load_image(real_ultrasound_path)
        _, metadata2 = processor.load_image(real_ultrasound_path)

        # Same file should produce same PHI-safe ID
        assert metadata1.phi_safe_id == metadata2.phi_safe_id

        # ID should not contain any path information
        assert str(real_ultrasound_path) not in metadata1.phi_safe_id
        assert "benign" not in metadata1.phi_safe_id.lower()
        assert "malignant" not in metadata1.phi_safe_id.lower()

    def test_warning_detection(self, processor):
        """Test that appropriate warnings are generated for problematic images."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT")

        warnings_found = set()

        # Sample images to find various warnings
        for img_path in list(breast_path.rglob("*.png"))[:30]:
            if "mask" in img_path.name.lower():
                continue

            _, metadata = processor.load_image(img_path)
            warnings_found.update(metadata.warnings)

        # We expect to find some warnings in a real medical dataset
        print(f"\nWarnings detected in real data: {warnings_found}")

        # Common warnings in ultrasound images
        possible_warnings = {
            "Low image quality detected",
            "Image appears to be very dark",
            "Image appears to be nearly blank",
            "Low entropy detected",
            "Low contrast detected"
        }

        # At least some warnings should be medical-relevant
        assert len(warnings_found) > 0 or len(warnings_found.intersection(possible_warnings)) > 0

    def test_mask_image_handling(self, processor):
        """Test that mask images are handled differently from ultrasound images."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign")

        if not breast_path.exists():
            pytest.skip("Dataset not found")

        # Find a mask image
        mask_paths = list(breast_path.glob("*_mask*.png"))
        if not mask_paths:
            pytest.skip("No mask images found")

        # Load mask - these are binary segmentation masks
        img, metadata = processor.load_image(mask_paths[0])

        # Masks typically have very different characteristics
        if img is not None:
            img_array = np.array(img)
            unique_vals = np.unique(img_array)

            # Masks are often binary or have very few unique values
            if len(unique_vals) < 10:
                print(f"\nMask detected with {len(unique_vals)} unique values")
                assert "Low entropy detected" in metadata.warnings or metadata.quality_score < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
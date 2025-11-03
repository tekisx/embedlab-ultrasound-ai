"""
Unit tests for PHI-safe logging utilities.

Tests ensure that:
1. PHI is properly sanitized from log messages
2. Hash-based identifiers are consistent
3. Performance metrics are tracked correctly
4. Log formatting works as expected
"""

import pytest
import logging
import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logging import (
    get_logger,
    phi_safe_identifier,
    PHISafeFormatter,
    MetricsLogger,
    log_image_processing
)


class TestPHISafeIdentifier:
    """Test PHI-safe identifier generation."""

    def test_consistent_hashing(self):
        """Test that same input produces same hash."""
        input_str = "patient_123_ultrasound.jpg"
        hash1 = phi_safe_identifier(input_str)
        hash2 = phi_safe_identifier(input_str)

        assert hash1 == hash2
        assert len(hash1) == 16  # First 16 chars of SHA-256

    def test_different_inputs_different_hashes(self):
        """Test that different inputs produce different hashes."""
        hash1 = phi_safe_identifier("image1.jpg")
        hash2 = phi_safe_identifier("image2.jpg")

        assert hash1 != hash2

    def test_path_input(self):
        """Test that Path objects are handled correctly."""
        path = Path("/patient/data/scan.jpg")
        hash_result = phi_safe_identifier(path)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 16

    def test_no_reversibility(self):
        """Test that hash cannot reveal original input."""
        original = "john_doe_123456_mri.dcm"
        hashed = phi_safe_identifier(original)

        # Hash should not contain any part of the original
        assert "john" not in hashed.lower()
        assert "doe" not in hashed.lower()
        assert "123456" not in hashed


class TestPHISafeFormatter:
    """Test PHI-safe log formatting."""

    def test_sanitize_file_paths(self):
        """Test that file paths are sanitized."""
        formatter = PHISafeFormatter(include_timestamp=False)

        # Test various file path formats
        test_cases = [
            ("Processing C:/patient_data/john_doe/ultrasound.jpg", "image_"),
            ("Loading /home/user/patient_123/scan.png", "image_"),
            ("Found file patient_456.dcm in directory", "image_"),
        ]

        for message, expected_prefix in test_cases:
            sanitized = formatter._sanitize_message(message)
            assert expected_prefix in sanitized
            assert "john_doe" not in sanitized
            assert "patient_123" not in sanitized
            assert "patient_456" not in sanitized

    def test_sanitize_numeric_ids(self):
        """Test that potential patient IDs are sanitized."""
        formatter = PHISafeFormatter(include_timestamp=False)

        message = "Patient ID: 123456789 processed successfully"
        sanitized = formatter._sanitize_message(message)

        assert "[ID_REDACTED]" in sanitized
        assert "123456789" not in sanitized

    def test_sanitize_dates(self):
        """Test that dates are sanitized."""
        formatter = PHISafeFormatter(include_timestamp=False)

        test_dates = [
            "DOB: 01/15/1980",
            "Scan date: 2024-03-15",
            "Visit on 12-25-2023",
        ]

        for message in test_dates:
            sanitized = formatter._sanitize_message(message)
            assert "[DATE_REDACTED]" in sanitized
            assert "1980" not in sanitized
            assert "2024" not in sanitized
            assert "2023" not in sanitized

    def test_preserve_safe_information(self):
        """Test that safe information is preserved."""
        formatter = PHISafeFormatter(include_timestamp=False)

        message = "Processed 100 images with 95% accuracy"
        sanitized = formatter._sanitize_message(message)

        assert "100" in sanitized  # Small numbers are safe
        assert "95%" in sanitized


class TestLogger:
    """Test logger creation and configuration."""

    def test_get_logger_basic(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_get_logger_with_level(self):
        """Test logger with custom level."""
        logger = get_logger("debug_logger", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_get_logger_with_file(self):
        """Test logger with file output."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = get_logger("file_logger", log_file=log_file)

            # Check that file handler was added
            file_handlers = [h for h in logger.handlers
                           if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) > 0

        finally:
            # Cleanup
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            if log_file.exists():
                log_file.unlink()

    def test_logger_phi_safety(self, caplog):
        """Test that logger sanitizes PHI in messages."""
        logger = get_logger("phi_test", level=logging.INFO)

        # Clear any existing handlers and add our test handler
        logger.handlers.clear()
        handler = logging.StreamHandler()
        handler.setFormatter(PHISafeFormatter(include_timestamp=False))
        logger.addHandler(handler)

        with caplog.at_level(logging.INFO):
            logger.info("Processing patient file: /data/john_doe_987654/ultrasound.jpg")

        # Check that PHI was sanitized
        assert "john_doe" not in caplog.text
        assert "987654" not in caplog.text
        assert "image_" in caplog.text


class TestMetricsLogger:
    """Test metrics logging functionality."""

    def test_log_operation(self):
        """Test operation logging."""
        base_logger = get_logger("metrics_test")
        metrics = MetricsLogger(base_logger)

        metrics.log_operation("embedding", 123.45, success=True)

        assert len(metrics.metrics['operations']) == 1
        op = metrics.metrics['operations'][0]
        assert op['operation'] == "embedding"
        assert op['duration_ms'] == 123.45
        assert op['success'] is True

    def test_log_performance(self):
        """Test performance metric logging."""
        base_logger = get_logger("perf_test")
        metrics = MetricsLogger(base_logger)

        metrics.log_performance("images_per_second", 25.5)

        assert metrics.metrics['performance']['images_per_second'] == 25.5

    def test_sanitize_details(self):
        """Test that operation details are sanitized."""
        base_logger = get_logger("sanitize_test")
        metrics = MetricsLogger(base_logger)

        details = {
            "file_path": "/patient/john_doe/scan.jpg",
            "size": 1024,
            "format": "JPEG"
        }

        metrics.log_operation("load", 10.0, success=True, details=details)

        op = metrics.metrics['operations'][0]
        # File path should be hashed
        assert "john_doe" not in str(op['details'])
        assert op['details']['size'] == 1024
        assert op['details']['format'] == "JPEG"

    def test_save_metrics(self):
        """Test saving metrics to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"

            base_logger = get_logger("save_test")
            metrics = MetricsLogger(base_logger)

            metrics.log_operation("test_op", 50.0)
            metrics.log_performance("test_metric", 100.0)

            metrics.save_metrics(output_path)

            # Verify file was created and contains expected data
            assert output_path.exists()

            with open(output_path) as f:
                saved_metrics = json.load(f)

            assert 'start_time' in saved_metrics
            assert 'end_time' in saved_metrics
            assert len(saved_metrics['operations']) == 1
            assert saved_metrics['performance']['test_metric'] == 100.0


class TestLogImageProcessing:
    """Test image processing logging helper."""

    def test_log_image_processing_basic(self, caplog):
        """Test basic image processing log."""
        logger = get_logger("img_test", level=logging.INFO)

        with caplog.at_level(logging.INFO):
            log_image_processing(logger, "patient_scan.jpg", "embedding")

        assert "Processing embedding for image_" in caplog.text
        assert "patient_scan" not in caplog.text

    def test_log_image_processing_with_metadata(self, caplog):
        """Test image processing log with metadata."""
        logger = get_logger("img_meta_test", level=logging.INFO)

        metadata = {
            "width": 512,
            "height": 512,
            "channels": 3,
            "patient_id": "12345",  # Should not appear in log
            "sensitive_info": "john_doe"  # Should not appear in log
        }

        with caplog.at_level(logging.INFO):
            log_image_processing(logger, "scan.jpg", "preprocessing", metadata)

        # Safe metadata should be included
        assert "512" in caplog.text
        assert "channels" in caplog.text

        # Sensitive metadata should not be included
        assert "12345" not in caplog.text
        assert "john_doe" not in caplog.text
        assert "patient_id" not in caplog.text


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
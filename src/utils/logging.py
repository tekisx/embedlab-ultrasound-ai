"""
PHI-safe logging utilities for medical imaging applications.

This module provides logging functionality that ensures no Protected Health
Information (PHI) is exposed in logs, using hash-based identifiers and
sanitized output formats.
"""

import logging
import hashlib
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json


class PHISafeFormatter(logging.Formatter):
    """Custom formatter that ensures PHI-safe logging."""

    def __init__(self, include_timestamp: bool = True):
        """
        Initialize the PHI-safe formatter.

        Args:
            include_timestamp: Whether to include timestamps in log messages
        """
        if include_timestamp:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            date_fmt = "%Y-%m-%d %H:%M:%S"
        else:
            format_str = "%(name)s - %(levelname)s - %(message)s"
            date_fmt = None

        super().__init__(format_str, datefmt=date_fmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with PHI safety checks.

        Args:
            record: The log record to format

        Returns:
            Formatted log message with PHI safety
        """
        # Sanitize the message to remove potential PHI
        if hasattr(record, 'msg'):
            record.msg = self._sanitize_message(str(record.msg))

        return super().format(record)

    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize message to remove potential PHI patterns.

        Args:
            message: Original message

        Returns:
            Sanitized message
        """
        # Remove common PHI patterns (simplified for this implementation)
        # In production, this would be more comprehensive

        # Replace file paths that might contain patient names with hashes
        import re

        # Pattern for file paths that might contain PHI
        # Matches: full paths (C:/path/file.ext or /path/file.ext) AND standalone filenames (file.ext)
        path_pattern = r'(?:(?:[A-Za-z]:[\\\/]|[\\\/])[^\\\/\s]*[\\\/])*([^\\\/\s]+\.(jpg|png|dcm|nii))'

        def replace_path(match):
            full_path = match.group(0)
            # Keep the extension but hash the rest
            if '.' in full_path:
                ext = full_path.split('.')[-1]
                hash_val = phi_safe_identifier(full_path)[:8]
                return f"image_{hash_val}.{ext}"
            return phi_safe_identifier(full_path)[:12]

        message = re.sub(path_pattern, replace_path, message, flags=re.IGNORECASE)

        # Remove potential patient IDs (numeric sequences > 5 digits)
        message = re.sub(r'\b\d{6,}\b', '[ID_REDACTED]', message)

        # Remove potential dates of birth (various formats)
        message = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE_REDACTED]', message)
        message = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DATE_REDACTED]', message)

        return message


def phi_safe_identifier(input_str: Union[str, Path]) -> str:
    """
    Generate a PHI-safe identifier from an input string.

    Uses SHA-256 hashing to create a consistent, anonymized identifier
    that cannot be reversed to obtain the original input.

    Args:
        input_str: Input string or path to hash

    Returns:
        Hexadecimal hash string (first 16 characters for brevity)

    Example:
        >>> phi_safe_identifier("patient_123_ultrasound.jpg")
        'a3f5c8d2b1e4f6a9'
    """
    if isinstance(input_str, Path):
        input_str = str(input_str)

    hash_obj = hashlib.sha256(input_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]


class MetricsLogger:
    """Logger for performance metrics and operational data."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.metrics: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'operations': [],
            'performance': {}
        }

    def log_operation(self,
                     operation: str,
                     duration_ms: float,
                     success: bool = True,
                     details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an operation with timing and success status.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            details: Additional details (PHI-safe only)
        """
        op_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }

        if details:
            # Ensure details are PHI-safe
            op_data['details'] = self._sanitize_details(details)

        self.metrics['operations'].append(op_data)

        # Also log to standard logger
        status = "completed" if success else "failed"
        self.logger.info(f"Operation '{operation}' {status} in {duration_ms:.2f}ms")

    def log_performance(self, metric_name: str, value: float) -> None:
        """
        Log a performance metric.

        Args:
            metric_name: Name of the metric (e.g., 'images_per_second')
            value: Metric value
        """
        self.metrics['performance'][metric_name] = value
        self.logger.info(f"Performance metric - {metric_name}: {value:.3f}")

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize details dictionary to ensure PHI safety.

        Args:
            details: Original details

        Returns:
            Sanitized details
        """
        sanitized = {}
        for key, value in details.items():
            if isinstance(value, (str, Path)):
                # Hash any string that might be a file path
                if any(ext in str(value) for ext in ['.jpg', '.png', '.dcm', '.nii']):
                    sanitized[key] = phi_safe_identifier(value)
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def save_metrics(self, output_path: Path) -> None:
        """
        Save metrics to JSON file.

        Args:
            output_path: Path to save metrics JSON
        """
        self.metrics['end_time'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

        self.logger.info(f"Metrics saved to {phi_safe_identifier(output_path)}")


def get_logger(name: str,
               level: Union[str, int] = logging.INFO,
               log_file: Optional[Path] = None,
               include_timestamp: bool = True) -> logging.Logger:
    """
    Get a configured logger with PHI-safe formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        include_timestamp: Whether to include timestamps

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        2024-01-01 12:00:00 - __main__ - INFO - Processing started
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler with PHI-safe formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(PHISafeFormatter(include_timestamp))
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(PHISafeFormatter(include_timestamp))
            logger.addHandler(file_handler)

    return logger


def log_image_processing(logger: logging.Logger,
                         image_path: Union[str, Path],
                         processing_type: str,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log image processing operations in a PHI-safe manner.

    Args:
        logger: Logger instance
        image_path: Path to image (will be hashed)
        processing_type: Type of processing performed
        metadata: Optional metadata (will be sanitized)

    Example:
        >>> log_image_processing(logger, "patient_123.jpg", "embedding")
        INFO - Processing embedding for image_a3f5c8d2.jpg
    """
    safe_id = phi_safe_identifier(image_path)

    # Extract safe metadata
    safe_metadata = {}
    if metadata:
        for key, value in metadata.items():
            if key in ['width', 'height', 'channels', 'format', 'size_kb']:
                safe_metadata[key] = value

    log_msg = f"Processing {processing_type} for image_{safe_id}"
    if safe_metadata:
        log_msg += f" - metadata: {safe_metadata}"

    logger.info(log_msg)


def log_embedding_generation(logger: logging.Logger,
                           image_path: Union[str, Path],
                           model_name: str,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log embedding generation operations in a PHI-safe manner.

    Args:
        logger: Logger instance
        image_path: Path to image (will be hashed)
        model_name: Name of the model used
        metadata: Optional metadata (will be sanitized)

    Example:
        >>> log_embedding_generation(logger, "patient_123.jpg", "resnet50", metadata)
        INFO - Generated embedding for image_a3f5c8d2 using resnet50
    """
    safe_id = phi_safe_identifier(image_path)

    # Extract safe metadata
    safe_metadata = {}
    if metadata:
        for key, value in metadata.items():
            if key in ['embedding_dim', 'generation_time_ms', 'is_cached']:
                safe_metadata[key] = value

    log_msg = f"Generated embedding for image_{safe_id} using {model_name}"
    if safe_metadata:
        log_msg += f" - metadata: {safe_metadata}"

    logger.info(log_msg)


# Module-level logger for this module
module_logger = get_logger(__name__)

if __name__ == "__main__":
    # Example usage
    logger = get_logger("test", level=logging.DEBUG)

    # Test PHI-safe logging
    logger.info("Starting image processing pipeline")
    logger.debug("Loading model weights")

    # This should be sanitized
    logger.info("Processing file C:/patient_data/john_doe_123456/ultrasound.jpg")

    # Test metrics logging
    metrics = MetricsLogger(logger)
    metrics.log_operation("image_load", 45.3, success=True)
    metrics.log_performance("images_per_second", 12.5)

    # Test image processing log
    log_image_processing(logger, "patient_123_ultrasound.jpg", "preprocessing",
                        {"width": 512, "height": 512, "channels": 3})
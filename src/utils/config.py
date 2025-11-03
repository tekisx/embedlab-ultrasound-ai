"""
Configuration and seed management for reproducible results.

This module ensures deterministic behavior across all operations,
critical for medical imaging validation and FDA compliance.
"""

import random
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

# Handle torch import gracefully
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import numpy as np

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    """
    Configuration settings for the Image Embedding Lab.

    All settings required for reproducible medical image processing.
    """

    # Seed settings for reproducibility
    seed: int = 42
    enable_deterministic: bool = True

    # Model settings
    model_name: str = "resnet50"
    pretrained: bool = True
    embedding_dim: int = 2048  # ResNet50 final layer dimension

    # Image processing settings
    image_size: int = 224
    batch_size: int = 16
    normalize_embeddings: bool = True
    image_channels: int = 3

    # Quality control settings
    min_image_size: int = 50
    max_image_size: int = 4000
    check_image_quality: bool = True
    reject_blank_images: bool = True
    blank_threshold: float = 0.05  # Threshold for blank detection (0-1)
    entropy_threshold: float = 1.0  # Minimum entropy for quality

    # Duplicate detection settings
    duplicate_threshold: float = 0.92
    use_union_find: bool = True

    # Anomaly detection settings
    anomaly_k_neighbors: int = 5
    anomaly_top_n: int = 8

    # Performance settings
    num_workers: int = 4
    use_memory_mapping: bool = True
    cache_embeddings: bool = True

    # Output settings
    output_format: str = "json"
    include_metadata: bool = True
    verbose: bool = True

    # PHI compliance settings
    use_phi_safe_logging: bool = True
    anonymize_filenames: bool = True

    # Storage paths (will be set during runtime)
    index_dir: Optional[Path] = None
    assets_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate numeric ranges
        if not 0 <= self.duplicate_threshold <= 1:
            raise ValueError(f"duplicate_threshold must be in [0, 1], got {self.duplicate_threshold}")

        if not 0 <= self.blank_threshold <= 1:
            raise ValueError(f"blank_threshold must be in [0, 1], got {self.blank_threshold}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.image_size < 32:
            raise ValueError(f"image_size must be >= 32, got {self.image_size}")

        # Convert string paths to Path objects
        if self.index_dir and not isinstance(self.index_dir, Path):
            self.index_dir = Path(self.index_dir)

        if self.assets_dir and not isinstance(self.assets_dir, Path):
            self.assets_dir = Path(self.assets_dir)

        if self.output_dir and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

        logger.info(f"Configuration initialized with seed={self.seed}")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save configuration

        Example:
            >>> config = Config(seed=123)
            >>> config.save("config.json")
        """
        path = Path(path)

        # Convert to dict and handle Path objects
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = value.as_posix()  # Always use forward slashes

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """
        Load configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            Config instance

        Example:
            >>> config = Config.load("config.json")
        """
        path = Path(path)

        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Convert string paths back to Path objects
        for key in ['index_dir', 'assets_dir', 'output_dir']:
            if key in config_dict and config_dict[key]:
                config_dict[key] = Path(config_dict[key])

        config = cls(**config_dict)
        logger.info(f"Configuration loaded from {path}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        config_dict = asdict(self)

        # Convert Path objects to strings with forward slashes for consistency
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = value.as_posix()  # Always use forward slashes

        return config_dict

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update

        Example:
            >>> config = Config()
            >>> config.update(seed=123, batch_size=32)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} = {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Re-validate
        self.__post_init__()


def set_all_seeds(seed: int, enable_deterministic: bool = True) -> None:
    """
    Set seeds for all random number generators.

    Ensures reproducibility across Python, NumPy, and PyTorch operations.
    Critical for clinical validation and regulatory compliance.

    Args:
        seed: Random seed value
        enable_deterministic: Enable deterministic algorithms (may impact performance)

    Example:
        >>> set_all_seeds(42)
        Setting all random seeds to 42
        Deterministic mode enabled for PyTorch
    """
    logger.info(f"Setting all random seeds to {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if enable_deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                    logger.info("Deterministic mode enabled for PyTorch")
                except Exception as e:
                    logger.warning(f"Could not enable full deterministic mode: {e}")
            else:
                logger.info("PyTorch deterministic mode partially enabled")
    else:
        logger.warning("PyTorch not available - seeds set for Python and NumPy only")

    # Environment variable for additional determinism
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.debug(f"PYTHONHASHSEED set to {seed}")


def create_reproducibility_manifest(config: Config, output_path: Path) -> Dict[str, Any]:
    """
    Create a manifest documenting the execution environment for reproducibility.

    Critical for FDA compliance and clinical validation.

    Args:
        config: Configuration instance
        output_path: Path to save manifest

    Returns:
        Manifest dictionary

    Example:
        >>> manifest = create_reproducibility_manifest(config, Path("manifest.json"))
    """
    import platform
    import sys
    from datetime import datetime

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {}
    }

    # Get package versions
    packages_to_check = ['numpy', 'torch', 'torchvision', 'Pillow', 'scikit-learn']

    for package in packages_to_check:
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                manifest['packages'][package] = module.__version__
        except ImportError:
            manifest['packages'][package] = "not installed"

    # Additional PyTorch information if available
    if TORCH_AVAILABLE:
        manifest['torch_details'] = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'deterministic': torch.backends.cudnn.deterministic if hasattr(torch.backends, 'cudnn') else None,
        }

    # Save manifest
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Reproducibility manifest created at {output_path}")

    return manifest


def get_default_config() -> Config:
    """
    Get default configuration optimized for medical imaging.

    Returns:
        Default Config instance

    Example:
        >>> config = get_default_config()
        >>> config.seed
        42
    """
    return Config()


def validate_hardware_requirements() -> Dict[str, Any]:
    """
    Validate that hardware meets minimum requirements.

    Returns:
        Dictionary with hardware validation results

    Example:
        >>> hw_check = validate_hardware_requirements()
        >>> if hw_check['meets_requirements']:
        ...     print("Hardware meets requirements")
    """
    import psutil

    validation = {
        'meets_requirements': True,
        'warnings': [],
        'info': {}
    }

    # Check RAM (minimum 4GB recommended)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    validation['info']['ram_gb'] = f"{ram_gb:.1f}"

    if ram_gb < 4:
        validation['warnings'].append(f"Low RAM: {ram_gb:.1f}GB (minimum 4GB recommended)")
        validation['meets_requirements'] = False

    # Check disk space (minimum 2GB free)
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024 ** 3)
    validation['info']['free_disk_gb'] = f"{free_gb:.1f}"

    if free_gb < 2:
        validation['warnings'].append(f"Low disk space: {free_gb:.1f}GB free")
        validation['meets_requirements'] = False

    # Check CPU cores
    cpu_count = psutil.cpu_count()
    validation['info']['cpu_cores'] = cpu_count

    if cpu_count < 2:
        validation['warnings'].append(f"Low CPU cores: {cpu_count} (multiple cores recommended)")

    # Check GPU if PyTorch available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        validation['info']['gpu_available'] = True
        validation['info']['gpu_name'] = torch.cuda.get_device_name(0)
        validation['info']['gpu_memory_gb'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}"
    else:
        validation['info']['gpu_available'] = False
        if TORCH_AVAILABLE:
            validation['warnings'].append("No GPU detected - processing will be slower")

    logger.info(f"Hardware validation: {'PASSED' if validation['meets_requirements'] else 'FAILED'}")

    for warning in validation['warnings']:
        logger.warning(warning)

    return validation


if __name__ == "__main__":
    # Example usage
    print("Testing configuration module...")

    # Create default config
    config = get_default_config()
    print(f"Default config created with seed={config.seed}")

    # Set seeds for reproducibility
    set_all_seeds(config.seed, config.enable_deterministic)

    # Validate hardware
    hw_check = validate_hardware_requirements()
    print(f"Hardware check: {hw_check}")

    # Save and load config
    config_path = Path("test_config.json")
    config.save(config_path)

    loaded_config = Config.load(config_path)
    print(f"Config loaded successfully: seed={loaded_config.seed}")

    # Create manifest
    manifest = create_reproducibility_manifest(config, Path("test_manifest.json"))
    print(f"Manifest created: {manifest['timestamp']}")

    # Clean up test files
    config_path.unlink()
    Path("test_manifest.json").unlink()
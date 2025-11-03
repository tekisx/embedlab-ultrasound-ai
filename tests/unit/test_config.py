"""
Unit tests for configuration and seed management.

Tests ensure that:
1. Seeds are properly set for reproducibility
2. Configuration can be saved/loaded
3. Parameter validation works correctly
4. Hardware requirements are checked
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.config import (
    Config,
    set_all_seeds,
    get_default_config,
    create_reproducibility_manifest,
    validate_hardware_requirements
)

# Check if torch is available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    import numpy as np
    TORCH_AVAILABLE = False


class TestConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()

        assert isinstance(config, Config)
        assert config.seed == 42
        assert config.model_name == "resnet50"
        assert config.batch_size == 16
        assert config.duplicate_threshold == 0.92

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Test invalid duplicate threshold
        with pytest.raises(ValueError, match="duplicate_threshold must be in"):
            Config(duplicate_threshold=1.5)

        # Test invalid blank threshold
        with pytest.raises(ValueError, match="blank_threshold must be in"):
            Config(blank_threshold=-0.1)

        # Test invalid batch size
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            Config(batch_size=0)

        # Test invalid image size
        with pytest.raises(ValueError, match="image_size must be >= 32"):
            Config(image_size=16)

    def test_config_update(self):
        """Test configuration update functionality."""
        config = Config()
        original_seed = config.seed

        config.update(seed=123, batch_size=32)

        assert config.seed == 123
        assert config.batch_size == 32

        # Test updating with invalid parameter (should warn but not fail)
        config.update(invalid_param=999)  # Should log warning

    def test_config_save_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.json"

            # Create and save config
            config = Config(seed=123, batch_size=32)
            config.index_dir = Path("/test/index")
            config.save(config_path)

            assert config_path.exists()

            # Load config
            loaded_config = Config.load(config_path)

            assert loaded_config.seed == 123
            assert loaded_config.batch_size == 32
            assert loaded_config.index_dir == Path("/test/index")

    def test_config_to_dict(self):
        """Test configuration dictionary conversion."""
        config = Config(seed=999)
        config.index_dir = Path("/test/path")

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['seed'] == 999
        assert config_dict['index_dir'] == "/test/path"  # Path converted to string

    def test_path_handling(self):
        """Test that Path objects are handled correctly."""
        config = Config(
            index_dir="/test/index",
            assets_dir="./assets",
            output_dir=Path("./output")
        )

        assert isinstance(config.index_dir, Path)
        assert isinstance(config.assets_dir, Path)
        assert isinstance(config.output_dir, Path)


class TestSeedManagement:
    """Test seed management for reproducibility."""

    def test_set_all_seeds_python(self):
        """Test that Python random seed is set correctly."""
        import random

        set_all_seeds(42)
        val1 = random.random()

        set_all_seeds(42)
        val2 = random.random()

        assert val1 == val2  # Should be identical with same seed

    def test_set_all_seeds_numpy(self):
        """Test that NumPy seed is set correctly."""
        set_all_seeds(42)
        arr1 = np.random.rand(5)

        set_all_seeds(42)
        arr2 = np.random.rand(5)

        np.testing.assert_array_equal(arr1, arr2)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_set_all_seeds_torch(self):
        """Test that PyTorch seed is set correctly."""
        set_all_seeds(42)
        tensor1 = torch.rand(5)

        set_all_seeds(42)
        tensor2 = torch.rand(5)

        assert torch.equal(tensor1, tensor2)

    def test_deterministic_mode(self):
        """Test that deterministic mode is enabled."""
        set_all_seeds(42, enable_deterministic=True)

        if TORCH_AVAILABLE:
            assert torch.backends.cudnn.deterministic == True
            assert torch.backends.cudnn.benchmark == False

        # Check environment variable
        import os
        assert os.environ.get('PYTHONHASHSEED') == '42'


class TestReproducibilityManifest:
    """Test reproducibility manifest creation."""

    def test_create_manifest(self):
        """Test manifest creation with all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            config = Config(seed=123)

            manifest = create_reproducibility_manifest(config, manifest_path)

            # Check manifest structure
            assert 'timestamp' in manifest
            assert 'config' in manifest
            assert 'environment' in manifest
            assert 'packages' in manifest

            # Check config is included
            assert manifest['config']['seed'] == 123

            # Check environment info
            assert 'python_version' in manifest['environment']
            assert 'platform' in manifest['environment']

            # Check package versions
            assert 'numpy' in manifest['packages']

            # Check file was created
            assert manifest_path.exists()

            # Verify JSON is valid
            with open(manifest_path) as f:
                loaded_manifest = json.load(f)
                assert loaded_manifest['config']['seed'] == 123

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_manifest_torch_details(self):
        """Test that PyTorch details are included in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            config = Config()

            manifest = create_reproducibility_manifest(config, manifest_path)

            assert 'torch_details' in manifest
            assert 'cuda_available' in manifest['torch_details']
            assert 'torch' in manifest['packages']


class TestHardwareValidation:
    """Test hardware requirements validation."""

    def test_validate_hardware(self):
        """Test basic hardware validation."""
        result = validate_hardware_requirements()

        assert isinstance(result, dict)
        assert 'meets_requirements' in result
        assert 'warnings' in result
        assert 'info' in result

        # Check that basic info is collected
        assert 'ram_gb' in result['info']
        assert 'free_disk_gb' in result['info']
        assert 'cpu_cores' in result['info']
        assert 'gpu_available' in result['info']

    def test_hardware_info_types(self):
        """Test that hardware info has correct types."""
        result = validate_hardware_requirements()

        # RAM and disk should be strings with numbers
        assert isinstance(result['info']['ram_gb'], str)
        assert float(result['info']['ram_gb']) > 0

        assert isinstance(result['info']['free_disk_gb'], str)
        assert float(result['info']['free_disk_gb']) >= 0

        # CPU cores should be integer
        assert isinstance(result['info']['cpu_cores'], int)
        assert result['info']['cpu_cores'] > 0


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create config
            config = Config(
                seed=999,
                batch_size=8,
                index_dir=tmpdir / "index",
                output_dir=tmpdir / "output"
            )

            # Set seeds
            set_all_seeds(config.seed, config.enable_deterministic)

            # Save config
            config_path = tmpdir / "config.json"
            config.save(config_path)

            # Create manifest
            manifest_path = tmpdir / "manifest.json"
            manifest = create_reproducibility_manifest(config, manifest_path)

            # Validate hardware
            hw_result = validate_hardware_requirements()

            # Verify everything worked
            assert config_path.exists()
            assert manifest_path.exists()
            assert hw_result['info']['cpu_cores'] > 0

            # Load and verify config
            loaded_config = Config.load(config_path)
            assert loaded_config.seed == 999
            assert loaded_config.batch_size == 8

    def test_config_with_all_paths(self):
        """Test configuration with all path fields set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create subdirectories
            (tmpdir / "index").mkdir()
            (tmpdir / "assets").mkdir()
            (tmpdir / "output").mkdir()

            config = Config(
                index_dir=tmpdir / "index",
                assets_dir=tmpdir / "assets",
                output_dir=tmpdir / "output"
            )

            # Save and reload
            config_path = tmpdir / "config.json"
            config.save(config_path)

            loaded = Config.load(config_path)

            assert loaded.index_dir == tmpdir / "index"
            assert loaded.assets_dir == tmpdir / "assets"
            assert loaded.output_dir == tmpdir / "output"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
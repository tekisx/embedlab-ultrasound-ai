"""
Unit tests for embedding generation module using REAL ultrasound images.

Tests deep learning embedding generation with actual medical data.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.embedding_engine import (
    EmbeddingEngine, EmbeddingMetadata, create_embedding_engine, TORCH_AVAILABLE
)
from src.utils.config import Config


# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class TestEmbeddingEngine:
    """Test suite for EmbeddingEngine using REAL ultrasound images."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = Config()
        config.seed = 42  # Ensure reproducibility
        return config

    @pytest.fixture
    def engine(self, config):
        """Create an embedding engine instance."""
        return EmbeddingEngine(config, model_name="resnet50")

    @pytest.fixture
    def real_ultrasound_images(self):
        """Get paths to real ultrasound images from dataset."""
        breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign")

        images = []
        if breast_path.exists():
            # Get up to 5 non-mask images
            for img_path in breast_path.glob("*.png"):
                if "mask" not in img_path.name.lower():
                    images.append(img_path)
                    if len(images) >= 5:
                        break

        if not images:
            pytest.skip("No real ultrasound images found. Run download_real_datasets.py first.")

        return images

    def test_engine_initialization(self, engine):
        """Test that the engine initializes correctly."""
        assert engine is not None
        assert engine.model is not None
        assert engine.embedding_dim == 2048  # ResNet50 dimension
        assert engine.device is not None

    def test_generate_single_embedding(self, engine, real_ultrasound_images):
        """Test generating embedding for a single real ultrasound image."""
        img_path = real_ultrasound_images[0]

        embedding, metadata = engine.generate_embedding(img_path)

        assert embedding is not None
        assert embedding.shape == (2048,)  # ResNet50 embedding dimension
        assert -1.1 <= embedding.min() <= embedding.max() <= 1.1  # Normalized

        # Check metadata
        assert metadata.embedding_dim == 2048
        assert metadata.model_name == "resnet50"
        assert metadata.generation_time_ms > 0
        assert metadata.image_hash is not None

    def test_embedding_normalization(self, engine, real_ultrasound_images):
        """Test that embeddings are L2 normalized."""
        img_path = real_ultrasound_images[0]

        embedding, _ = engine.generate_embedding(img_path)

        if embedding is not None:
            # Check L2 norm is approximately 1
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"

    def test_embedding_caching(self, engine, real_ultrasound_images):
        """Test that embedding caching works correctly."""
        img_path = real_ultrasound_images[0]

        # First generation
        embedding1, metadata1 = engine.generate_embedding(img_path)
        assert not metadata1.is_cached

        # Second generation should be cached
        embedding2, metadata2 = engine.generate_embedding(img_path)
        assert metadata2.is_cached
        assert metadata2.generation_time_ms < metadata1.generation_time_ms

        # Embeddings should be identical
        if embedding1 is not None and embedding2 is not None:
            np.testing.assert_array_almost_equal(embedding1, embedding2)

    def test_batch_embedding_generation(self, engine, real_ultrasound_images):
        """Test batch generation of embeddings."""
        batch_size = min(3, len(real_ultrasound_images))
        batch_paths = real_ultrasound_images[:batch_size]

        embeddings, metadatas = engine.generate_batch_embeddings(batch_paths, batch_size=2)

        assert embeddings.shape == (batch_size, 2048)
        assert len(metadatas) == batch_size

        # Check each embedding is normalized
        for i in range(batch_size):
            norm = np.linalg.norm(embeddings[i])
            assert abs(norm - 1.0) < 0.01 or norm == 0  # Either normalized or zero (failed)

    def test_similarity_computation(self, engine, real_ultrasound_images):
        """Test cosine similarity computation between embeddings."""
        if len(real_ultrasound_images) < 2:
            pytest.skip("Need at least 2 images for similarity test")

        embedding1, _ = engine.generate_embedding(real_ultrasound_images[0])
        embedding2, _ = engine.generate_embedding(real_ultrasound_images[1])

        if embedding1 is not None and embedding2 is not None:
            similarity = engine.compute_similarity(embedding1, embedding2)

            assert -1.0 <= similarity <= 1.0
            # Different ultrasound images should have moderate similarity
            assert 0.3 <= similarity <= 0.95

    def test_self_similarity(self, engine, real_ultrasound_images):
        """Test that an image has perfect similarity with itself."""
        img_path = real_ultrasound_images[0]

        embedding, _ = engine.generate_embedding(img_path)

        if embedding is not None:
            similarity = engine.compute_similarity(embedding, embedding)
            assert abs(similarity - 1.0) < 0.001  # Should be exactly 1

    def test_find_similar_images(self, engine, real_ultrasound_images):
        """Test finding similar images in a database."""
        if len(real_ultrasound_images) < 3:
            pytest.skip("Need at least 3 images for similarity search test")

        # Generate database embeddings
        db_embeddings = []
        for img_path in real_ultrasound_images[:3]:
            embedding, _ = engine.generate_embedding(img_path)
            if embedding is not None:
                db_embeddings.append(embedding)

        if len(db_embeddings) < 2:
            pytest.skip("Not enough valid embeddings generated")

        db_array = np.stack(db_embeddings)

        # Use first image as query
        query_embedding = db_embeddings[0]

        # Find similar images
        similar = engine.find_similar_images(
            query_embedding, db_array, top_k=2, threshold=0.5
        )

        assert len(similar) > 0
        # First result should be the query itself with similarity ~1
        assert similar[0][0] == 0  # Index 0
        assert similar[0][1] > 0.99  # Similarity ~1

    def test_save_and_load_embeddings(self, engine, real_ultrasound_images):
        """Test saving and loading embeddings to/from disk."""
        # Generate embeddings
        embeddings_list = []
        metadatas = []

        for img_path in real_ultrasound_images[:2]:
            embedding, metadata = engine.generate_embedding(img_path)
            if embedding is not None:
                embeddings_list.append(embedding)
                metadatas.append(metadata)

        if not embeddings_list:
            pytest.skip("No embeddings generated")

        embeddings_array = np.stack(embeddings_list)

        # Save embeddings
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_embeddings.npy"
            engine.save_embeddings(embeddings_array, save_path, metadatas)

            assert save_path.exists()
            assert save_path.with_suffix('.json').exists()  # Metadata file

            # Load embeddings
            loaded_embeddings, loaded_metadata = engine.load_embeddings(save_path)

            assert loaded_embeddings is not None
            np.testing.assert_array_almost_equal(embeddings_array, loaded_embeddings)
            assert loaded_metadata is not None
            assert len(loaded_metadata) == len(metadatas)

    def test_different_models(self, config):
        """Test that different models produce different embedding dimensions."""
        models_to_test = {
            "resnet50": 2048,
            "resnet18": 512,
        }

        for model_name, expected_dim in models_to_test.items():
            engine = EmbeddingEngine(config, model_name=model_name)
            assert engine.embedding_dim == expected_dim

    def test_reproducibility(self, config, real_ultrasound_images):
        """Test that embeddings are reproducible with same seed."""
        img_path = real_ultrasound_images[0]

        # Create two engines with same config (same seed)
        engine1 = EmbeddingEngine(config, model_name="resnet50")
        engine2 = EmbeddingEngine(config, model_name="resnet50")

        embedding1, _ = engine1.generate_embedding(img_path)
        embedding2, _ = engine2.generate_embedding(img_path)

        if embedding1 is not None and embedding2 is not None:
            # Embeddings should be identical with same seed
            np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)

    def test_cache_management(self, engine, real_ultrasound_images):
        """Test cache clearing and info retrieval."""
        # Generate some cached embeddings
        for img_path in real_ultrasound_images[:2]:
            engine.generate_embedding(img_path)

        # Check cache info
        cache_info = engine.get_cache_info()
        assert cache_info['cache_size'] == 2
        assert cache_info['cache_memory_mb'] > 0

        # Clear cache
        engine.clear_cache()
        cache_info = engine.get_cache_info()
        assert cache_info['cache_size'] == 0

    def test_invalid_image_handling(self, engine):
        """Test handling of invalid image paths."""
        invalid_path = Path("nonexistent_image.png")

        embedding, metadata = engine.generate_embedding(invalid_path)

        assert embedding is None
        assert metadata.embedding_dim == 2048
        assert metadata.generation_time_ms > 0

    def test_factory_function(self, config):
        """Test the factory function for creating engines."""
        engine = create_embedding_engine(config, model_name="resnet50")

        assert engine is not None
        assert isinstance(engine, EmbeddingEngine)
        assert engine.embedding_dim == 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
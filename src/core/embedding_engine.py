"""
Deep learning embedding generation for medical images using ResNet50.

This module provides embedding generation using pretrained models,
optimized for medical imaging applications.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
import time
import hashlib
from dataclasses import dataclass

from ..utils.logging import get_logger, phi_safe_identifier, log_embedding_generation
from ..utils.config import Config, set_all_seeds
from .image_processor import ImageProcessor

logger = get_logger(__name__)

# Try to import torch, but make it optional
try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some features will be disabled.")


@dataclass
class EmbeddingMetadata:
    """Metadata for generated embeddings."""

    image_hash: str
    embedding_dim: int
    model_name: str
    generation_time_ms: float
    is_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'image_hash': self.image_hash,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'generation_time_ms': round(self.generation_time_ms, 2),
            'is_cached': self.is_cached
        }


class EmbeddingEngine:
    """
    Generate embeddings from medical images using deep learning models.

    Designed for ultrasound imaging with reproducibility and efficiency.
    """

    def __init__(self,
                 config: Optional[Config] = None,
                 model_name: str = "resnet50",
                 cache_embeddings: bool = True):
        """
        Initialize the embedding engine.

        Args:
            config: Configuration object
            model_name: Name of the model to use
            cache_embeddings: Whether to cache generated embeddings
        """
        self.config = config or Config()
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings

        # Set seeds for reproducibility
        set_all_seeds(self.config.seed)

        # Initialize image processor
        self.image_processor = ImageProcessor(self.config)

        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Initialize model
        self.model = None
        self.device = None
        self.embedding_dim = 0

        if TORCH_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("PyTorch not available. Cannot initialize embedding model.")

    def _initialize_model(self):
        """Initialize the deep learning model for embedding generation."""
        try:
            # Set device (GPU if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load pretrained model
            if self.model_name == "resnet50":
                # Load ResNet50 pretrained on ImageNet
                base_model = models.resnet50(pretrained=True)

                # Remove the final classification layer to get features
                # ResNet50's avgpool output is 2048-dimensional
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 2048

            elif self.model_name == "resnet18":
                # Lighter alternative for faster processing
                base_model = models.resnet18(pretrained=True)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 512

            elif self.model_name == "efficientnet_b0":
                # EfficientNet for better efficiency
                base_model = models.efficientnet_b0(pretrained=True)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 1280

            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Disable gradient computation for inference
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info(f"Initialized {self.model_name} with {self.embedding_dim}-dim embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def generate_embedding(self,
                          image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], EmbeddingMetadata]:
        """
        Generate embedding for a single image.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (embedding array or None, metadata)
        """
        start_time = time.time()
        image_path = Path(image_path)
        image_hash = phi_safe_identifier(image_path)

        metadata = EmbeddingMetadata(
            image_hash=image_hash,
            embedding_dim=self.embedding_dim,
            model_name=self.model_name,
            generation_time_ms=0
        )

        # Check cache first
        if self.cache_embeddings and image_hash in self._embedding_cache:
            metadata.is_cached = True
            metadata.generation_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Retrieved cached embedding for {image_hash}")
            return self._embedding_cache[image_hash].copy(), metadata

        if not TORCH_AVAILABLE or self.model is None:
            logger.error("Model not available for embedding generation")
            metadata.generation_time_ms = (time.time() - start_time) * 1000
            return None, metadata

        try:
            # Load and preprocess image
            img, img_metadata = self.image_processor.load_image(image_path)

            if img is None or not img_metadata.is_valid:
                logger.warning(f"Invalid image for embedding: {image_hash}")
                metadata.generation_time_ms = (time.time() - start_time) * 1000
                return None, metadata

            # Preprocess for model input
            img_tensor = self.image_processor.preprocess_for_embedding(img)

            # Convert to torch tensor and move to device
            img_tensor = torch.from_numpy(img_tensor).to(self.device)

            # Generate embedding
            with torch.no_grad():
                embedding_tensor = self.model(img_tensor)

            # Convert to numpy and flatten
            embedding = embedding_tensor.cpu().numpy().flatten()

            # Normalize embedding (L2 normalization for cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Cache if enabled
            if self.cache_embeddings:
                self._embedding_cache[image_hash] = embedding.copy()

            metadata.generation_time_ms = (time.time() - start_time) * 1000

            log_embedding_generation(logger, image_path, self.model_name, metadata.to_dict())

            return embedding, metadata

        except Exception as e:
            logger.error(f"Failed to generate embedding for {image_hash}: {e}")
            metadata.generation_time_ms = (time.time() - start_time) * 1000
            return None, metadata

    def generate_batch_embeddings(self,
                                 image_paths: List[Union[str, Path]],
                                 batch_size: int = 32) -> Tuple[np.ndarray, List[EmbeddingMetadata]]:
        """
        Generate embeddings for multiple images in batches.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing

        Returns:
            Tuple of (embeddings array (N, D), list of metadata)
        """
        embeddings = []
        metadata_list = []

        # Process in batches for memory efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

            for path in batch_paths:
                embedding, metadata = self.generate_embedding(path)

                if embedding is not None:
                    embeddings.append(embedding)
                    metadata_list.append(metadata)
                else:
                    # Add zero embedding for failed images to maintain alignment
                    embeddings.append(np.zeros(self.embedding_dim))
                    metadata_list.append(metadata)

        if embeddings:
            embeddings_array = np.stack(embeddings, axis=0)
            logger.info(f"Generated embeddings for {len(embeddings)} images")
            return embeddings_array, metadata_list
        else:
            logger.warning("No embeddings generated")
            return np.array([]), metadata_list

    def compute_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2

        # Compute cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)

        return float(similarity)

    def find_similar_images(self,
                           query_embedding: np.ndarray,
                           database_embeddings: np.ndarray,
                           top_k: int = 5,
                           threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Find similar images based on embedding similarity.

        Args:
            query_embedding: Query image embedding
            database_embeddings: Database of embeddings (N, D)
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples
        """
        if len(database_embeddings) == 0:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_normalized = query_embedding / query_norm

        # Compute similarities with all database embeddings
        similarities = []
        for idx, db_embedding in enumerate(database_embeddings):
            similarity = self.compute_similarity(query_normalized, db_embedding)
            if similarity >= threshold:
                similarities.append((idx, similarity))

        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def save_embeddings(self,
                       embeddings: np.ndarray,
                       output_path: Union[str, Path],
                       metadata: Optional[List[EmbeddingMetadata]] = None):
        """
        Save embeddings to disk in numpy format.

        Args:
            embeddings: Embeddings array
            output_path: Path to save embeddings
            metadata: Optional metadata to save alongside
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save embeddings
            np.save(output_path, embeddings)
            logger.info(f"Saved embeddings to {output_path}")

            # Save metadata if provided
            if metadata:
                import json
                metadata_path = output_path.with_suffix('.json')
                metadata_dicts = [m.to_dict() for m in metadata]

                with open(metadata_path, 'w') as f:
                    json.dump(metadata_dicts, f, indent=2)
                logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise

    def load_embeddings(self,
                       embeddings_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        Load embeddings from disk.

        Args:
            embeddings_path: Path to embeddings file

        Returns:
            Tuple of (embeddings array or None, metadata list or None)
        """
        embeddings_path = Path(embeddings_path)

        try:
            # Load embeddings
            embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings from {embeddings_path}")

            # Try to load metadata
            metadata = None
            metadata_path = embeddings_path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")

            return embeddings, metadata

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None, None

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Cleared embedding cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            'cache_size': len(self._embedding_cache),
            'cached_images': list(self._embedding_cache.keys()),
            'cache_memory_mb': sum(e.nbytes for e in self._embedding_cache.values()) / (1024 * 1024)
        }


def create_embedding_engine(config: Optional[Config] = None,
                          model_name: str = "resnet50") -> Optional[EmbeddingEngine]:
    """
    Factory function to create an embedding engine.

    Args:
        config: Configuration object
        model_name: Model to use for embeddings

    Returns:
        EmbeddingEngine instance or None if initialization fails
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for embedding generation")
        return None

    try:
        engine = EmbeddingEngine(config, model_name)
        return engine
    except Exception as e:
        logger.error(f"Failed to create embedding engine: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    config = Config()
    engine = create_embedding_engine(config)

    if engine:
        # Test with a sample ultrasound image
        test_image = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign/benign (1).png")
        if test_image.exists():
            embedding, metadata = engine.generate_embedding(test_image)
            if embedding is not None:
                print(f"Generated embedding with shape: {embedding.shape}")
                print(f"Metadata: {metadata.to_dict()}")
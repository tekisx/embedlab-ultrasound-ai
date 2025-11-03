"""
Central index manager for the embedding pipeline.

Coordinates image processing, embedding generation, similarity search,
and duplicate detection for medical ultrasound images.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json
import time
from datetime import datetime
import hashlib

from ..utils.logging import get_logger, phi_safe_identifier, MetricsLogger
from ..utils.config import Config
from .image_processor import ImageProcessor
from .embedding_engine import EmbeddingEngine, create_embedding_engine
from .similarity_search import SimilaritySearch

logger = get_logger(__name__)


class IndexManager:
    """
    Manages the complete embedding pipeline for medical images.

    Coordinates all components and provides a unified interface.
    """

    def __init__(self,
                 config: Optional[Config] = None,
                 index_path: Optional[Union[str, Path]] = None):
        """
        Initialize the index manager.

        Args:
            config: Configuration object
            index_path: Path to store/load index
        """
        self.config = config or Config()
        self.index_path = Path(index_path) if index_path else Path("index")

        # Initialize components
        self.image_processor = ImageProcessor(self.config)
        self.embedding_engine = create_embedding_engine(self.config)
        self.similarity_search = SimilaritySearch(self.config)

        # Metrics tracking
        self.metrics_logger = MetricsLogger(logger)

        # Image registry
        self.image_registry: Dict[str, Dict] = {}

        logger.info(f"IndexManager initialized with index at {self.index_path}")

    def add_image(self,
                 image_path: Union[str, Path],
                 generate_embedding: bool = True) -> Dict:
        """
        Add a single image to the index.

        Args:
            image_path: Path to image
            generate_embedding: Whether to generate embedding

        Returns:
            Result dictionary with status and metadata
        """
        start_time = time.time()
        image_path = Path(image_path)
        result = {
            'success': False,
            'image_hash': None,
            'message': '',
            'metadata': {}
        }

        try:
            # Load and validate image
            img, img_metadata = self.image_processor.load_image(image_path)

            if img is None or not img_metadata.is_valid:
                result['message'] = f"Invalid image: {img_metadata.warnings}"
                return result

            image_hash = img_metadata.phi_safe_id
            result['image_hash'] = image_hash

            # Check if already in index
            if image_hash in self.image_registry:
                result['message'] = "Image already in index"
                result['success'] = True
                return result

            # Generate embedding if requested
            embedding = None
            if generate_embedding and self.embedding_engine:
                embedding, emb_metadata = self.embedding_engine.generate_embedding(image_path)

                if embedding is not None:
                    # Add to similarity search
                    self.similarity_search.add_embeddings(
                        np.expand_dims(embedding, axis=0),
                        [image_hash],
                        [img_metadata.to_safe_dict()]
                    )

            # Add to registry
            self.image_registry[image_hash] = {
                'original_path': str(image_path),
                'added_at': datetime.now().isoformat(),
                'image_metadata': img_metadata.to_safe_dict(),
                'has_embedding': embedding is not None
            }

            # Log metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics_logger.log_operation(
                "add_image",
                duration_ms,
                success=True,
                details={'image_hash': image_hash}
            )

            result['success'] = True
            result['message'] = "Image added successfully"
            result['metadata'] = img_metadata.to_safe_dict()

        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            result['message'] = f"Error: {str(e)}"

        return result

    def add_batch(self,
                 image_paths: List[Union[str, Path]],
                 batch_size: int = 32,
                 generate_embeddings: bool = True) -> Dict:
        """
        Add multiple images in batch.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            generate_embeddings: Whether to generate embeddings

        Returns:
            Summary dictionary
        """
        start_time = time.time()
        results = {
            'total': len(image_paths),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

        logger.info(f"Adding batch of {len(image_paths)} images")

        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}")

            # Load images
            images, metadatas = self.image_processor.load_batch(batch_paths)

            # Generate embeddings if requested
            if generate_embeddings and self.embedding_engine:
                valid_paths = [
                    path for path, img in zip(batch_paths, images)
                    if img is not None
                ]

                if valid_paths:
                    embeddings, emb_metadatas = self.embedding_engine.generate_batch_embeddings(
                        valid_paths, batch_size=min(batch_size, 16)
                    )

                    # Add to similarity search
                    valid_hashes = []
                    valid_embeddings = []
                    valid_metadata = []

                    for idx, (embedding, metadata) in enumerate(zip(embeddings, emb_metadatas)):
                        if np.linalg.norm(embedding) > 0:  # Valid embedding
                            valid_hashes.append(metadata.image_hash)
                            valid_embeddings.append(embedding)
                            valid_metadata.append(metadatas[idx].to_safe_dict())

                    if valid_embeddings:
                        self.similarity_search.add_embeddings(
                            np.stack(valid_embeddings),
                            valid_hashes,
                            valid_metadata
                        )

            # Update registry and results
            for path, img, metadata in zip(batch_paths, images, metadatas):
                if img is not None and metadata.is_valid:
                    image_hash = metadata.phi_safe_id

                    if image_hash not in self.image_registry:
                        self.image_registry[image_hash] = {
                            'original_path': str(path),
                            'added_at': datetime.now().isoformat(),
                            'image_metadata': metadata.to_safe_dict(),
                            'has_embedding': generate_embeddings
                        }
                        results['successful'] += 1
                    else:
                        results['skipped'] += 1
                else:
                    results['failed'] += 1
                    if metadata.warnings:
                        results['errors'].append({
                            'path': phi_safe_identifier(path),
                            'warnings': metadata.warnings
                        })

        # Build search index
        if generate_embeddings:
            self.similarity_search.build_index()

        # Log metrics
        duration_ms = (time.time() - start_time) * 1000
        self.metrics_logger.log_operation(
            "add_batch",
            duration_ms,
            success=True,
            details={'total': results['total'], 'successful': results['successful']}
        )

        logger.info(f"Batch processing complete: {results}")
        return results

    def search(self,
              query_path: Union[str, Path],
              top_k: int = 5,
              threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar images.

        Args:
            query_path: Path to query image
            top_k: Number of results
            threshold: Similarity threshold

        Returns:
            List of search results
        """
        start_time = time.time()
        query_path = Path(query_path)

        # Load query image
        img, metadata = self.image_processor.load_image(query_path)
        if img is None:
            logger.error(f"Failed to load query image: {query_path}")
            return []

        # Generate embedding
        if self.embedding_engine is None:
            logger.error("Embedding engine not available")
            return []

        embedding, _ = self.embedding_engine.generate_embedding(query_path)
        if embedding is None:
            logger.error("Failed to generate query embedding")
            return []

        # Search
        search_results = self.similarity_search.search(
            embedding, top_k=top_k, threshold=threshold
        )

        # Enhance results with registry data
        enhanced_results = []
        for result in search_results:
            enhanced = result.to_dict()

            if result.image_hash in self.image_registry:
                registry_data = self.image_registry[result.image_hash]
                enhanced['image_info'] = {
                    'added_at': registry_data['added_at'],
                    'quality_score': registry_data['image_metadata'].get('quality_score', 0)
                }

            enhanced_results.append(enhanced)

        # Log metrics
        duration_ms = (time.time() - start_time) * 1000
        self.metrics_logger.log_operation(
            "search",
            duration_ms,
            success=True,
            details={'n_results': len(enhanced_results)}
        )

        return enhanced_results

    def find_duplicates(self, threshold: float = 0.95) -> List[Dict]:
        """
        Find potential duplicate images in the index.

        Args:
            threshold: Similarity threshold for duplicates

        Returns:
            List of duplicate pairs
        """
        duplicates = self.similarity_search.find_duplicates(threshold)

        # Enhance with registry data
        enhanced_duplicates = []
        for hash1, hash2, similarity in duplicates:
            enhanced = {
                'image1_hash': hash1,
                'image2_hash': hash2,
                'similarity': similarity
            }

            # Add quality scores if available
            if hash1 in self.image_registry:
                enhanced['image1_quality'] = self.image_registry[hash1]['image_metadata'].get('quality_score', 0)
            if hash2 in self.image_registry:
                enhanced['image2_quality'] = self.image_registry[hash2]['image_metadata'].get('quality_score', 0)

            enhanced_duplicates.append(enhanced)

        logger.info(f"Found {len(enhanced_duplicates)} duplicate pairs")
        return enhanced_duplicates

    def remove_image(self, image_hash: str) -> bool:
        """
        Remove an image from the index.

        Args:
            image_hash: Hash of image to remove

        Returns:
            True if removed successfully
        """
        if image_hash not in self.image_registry:
            return False

        # Remove from similarity search
        self.similarity_search.remove_embedding(image_hash)

        # Remove from registry
        del self.image_registry[image_hash]

        # Clear embedding cache if exists
        if self.embedding_engine:
            cache_info = self.embedding_engine.get_cache_info()
            if image_hash in cache_info.get('cached_images', []):
                self.embedding_engine.clear_cache()

        logger.info(f"Removed image {image_hash} from index")
        return True

    def save_index(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the complete index to disk.

        Args:
            output_path: Path to save index (uses self.index_path if None)
        """
        output_path = Path(output_path) if output_path else self.index_path
        output_path.mkdir(parents=True, exist_ok=True)

        # Save similarity search index
        self.similarity_search.save_index(output_path / "embeddings")

        # Save registry
        registry_path = output_path / "registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.image_registry, f, indent=2)

        # Save metrics
        self.metrics_logger.save_metrics(output_path / "metrics.json")

        # Save config
        config_path = output_path / "config.json"
        self.config.save(config_path)

        logger.info(f"Index saved to {output_path}")

    def load_index(self, index_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load index from disk.

        Args:
            index_path: Path to index (uses self.index_path if None)
        """
        index_path = Path(index_path) if index_path else self.index_path

        if not index_path.exists():
            logger.error(f"Index path does not exist: {index_path}")
            return

        # Load similarity search index
        self.similarity_search.load_index(index_path / "embeddings")

        # Load registry
        registry_path = index_path / "registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.image_registry = json.load(f)
            logger.info(f"Loaded registry with {len(self.image_registry)} images")

        # Load config
        config_path = index_path / "config.json"
        if config_path.exists():
            self.config = Config.load(config_path)
            logger.info("Loaded configuration from index")

    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the index.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_images': len(self.image_registry),
            'n_embeddings': self.similarity_search.embeddings.shape[0]
                           if self.similarity_search.embeddings is not None else 0,
            'index_path': str(self.index_path),
            'similarity_search': self.similarity_search.get_statistics()
        }

        # Add quality distribution
        if self.image_registry:
            quality_scores = [
                reg['image_metadata'].get('quality_score', 0)
                for reg in self.image_registry.values()
            ]
            stats['quality_distribution'] = {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            }

        # Add cache info if available
        if self.embedding_engine:
            stats['embedding_cache'] = self.embedding_engine.get_cache_info()

        return stats

    def validate_index(self) -> Dict:
        """
        Validate the integrity of the index.

        Returns:
            Validation results
        """
        issues = []

        # Check registry vs embeddings consistency
        registry_hashes = set(self.image_registry.keys())
        if self.similarity_search.image_hashes:
            search_hashes = set(self.similarity_search.image_hashes)

            # Find mismatches
            only_in_registry = registry_hashes - search_hashes
            only_in_search = search_hashes - registry_hashes

            if only_in_registry:
                issues.append(f"Images in registry but not in search: {len(only_in_registry)}")
            if only_in_search:
                issues.append(f"Images in search but not in registry: {len(only_in_search)}")

        # Check for duplicate hashes
        if self.similarity_search.image_hashes:
            if len(self.similarity_search.image_hashes) != len(set(self.similarity_search.image_hashes)):
                issues.append("Duplicate hashes in search index")

        validation_result = {
            'valid': len(issues) == 0,
            'issues': issues,
            'n_images': len(self.image_registry),
            'n_embeddings': len(self.similarity_search.image_hashes)
                          if self.similarity_search.image_hashes else 0
        }

        if validation_result['valid']:
            logger.info("Index validation passed")
        else:
            logger.warning(f"Index validation found issues: {issues}")

        return validation_result

    def get_image_path(self, image_hash: str) -> Optional[str]:
        """
        Get original image path for a given hash.

        Args:
            image_hash: The image hash

        Returns:
            Original image path or None if not found
        """
        if image_hash in self.image_registry:
            return self.image_registry[image_hash].get('original_path')
        return None


def create_index_manager(config: Optional[Config] = None,
                        index_path: Optional[Union[str, Path]] = None) -> IndexManager:
    """
    Factory function to create an index manager.

    Args:
        config: Configuration object
        index_path: Path to index

    Returns:
        IndexManager instance
    """
    return IndexManager(config, index_path)


if __name__ == "__main__":
    # Example usage with real ultrasound images
    config = Config()
    manager = create_index_manager(config, "ultrasound_index")

    # Add sample images from dataset
    breast_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign")
    if breast_path.exists():
        image_paths = list(breast_path.glob("*.png"))[:5]
        results = manager.add_batch(image_paths)
        print(f"Added {results['successful']} images")

        # Search for similar
        if image_paths:
            search_results = manager.search(image_paths[0])
            print(f"Found {len(search_results)} similar images")

        # Save index
        manager.save_index()
        print("Index saved successfully")
"""
Fast similarity search for medical image embeddings.

This module provides efficient nearest-neighbor search using cosine similarity
and optimized index structures.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import time
from dataclasses import dataclass
import json

from ..utils.logging import get_logger, phi_safe_identifier
from ..utils.config import Config

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result from a similarity search."""

    index: int
    similarity: float
    image_hash: str
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'index': self.index,
            'similarity': round(self.similarity, 4),
            'image_hash': self.image_hash,
            'metadata': self.metadata
        }


class SimilaritySearch:
    """
    Fast similarity search engine for image embeddings.

    Optimized for medical imaging with support for large-scale searches.
    """

    def __init__(self,
                 config: Optional[Config] = None,
                 distance_metric: str = "cosine"):
        """
        Initialize similarity search engine.

        Args:
            config: Configuration object
            distance_metric: Distance metric to use ("cosine", "euclidean")
        """
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"Invalid distance metric: {distance_metric}. Must be one of {valid_metrics}")

        self.config = config or Config()
        self.distance_metric = distance_metric

        # Storage for embeddings and metadata
        self.embeddings: Optional[np.ndarray] = None
        self.image_hashes: List[str] = []
        self.metadata: List[Dict] = []

        # Index for fast search (can be extended with FAISS, Annoy, etc.)
        self.index_built = False

        logger.info(f"SimilaritySearch initialized with {distance_metric} metric")

    def add_embeddings(self,
                      embeddings: np.ndarray,
                      image_hashes: List[str],
                      metadata: Optional[List[Dict]] = None) -> None:
        """
        Add embeddings to the search index.

        Args:
            embeddings: Array of embeddings (N, D)
            image_hashes: List of image hashes
            metadata: Optional metadata for each embedding
        """
        if len(embeddings) != len(image_hashes):
            raise ValueError("Number of embeddings must match number of hashes")

        # Normalize embeddings for cosine similarity
        if self.distance_metric == "cosine":
            embeddings = self._normalize_embeddings(embeddings)

        if self.embeddings is None:
            self.embeddings = embeddings
            self.image_hashes = image_hashes
            self.metadata = metadata or [{}] * len(image_hashes)
        else:
            # Append to existing
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.image_hashes.extend(image_hashes)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(image_hashes))

        self.index_built = False  # Need to rebuild index
        logger.info(f"Added {len(embeddings)} embeddings. Total: {len(self.embeddings)}")

    def build_index(self) -> None:
        """Build search index for fast retrieval."""
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings to index")
            return

        # For now, we use brute-force search
        # In production, would use FAISS, Annoy, or similar
        self.index_built = True
        logger.info(f"Built index for {len(self.embeddings)} embeddings")

    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings in index")
            return []

        start_time = time.time()

        # Normalize query if using cosine similarity
        if self.distance_metric == "cosine":
            query_embedding = self._normalize_embedding(query_embedding)

        # Compute similarities
        if self.distance_metric == "cosine":
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
        elif self.distance_metric == "euclidean":
            # Convert distances to similarities
            distances = self._euclidean_distance(query_embedding, self.embeddings)
            # Normalize to 0-1 range (1 = identical, 0 = very different)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            similarities = 1.0 - (distances / max_dist) if max_dist > 0 else distances
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # Apply threshold if specified
        if threshold is not None:
            valid_indices = np.where(similarities >= threshold)[0]
        else:
            valid_indices = np.arange(len(similarities))

        # Get top-k results
        if len(valid_indices) > 0:
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
        else:
            top_indices = []

        # Create results
        results = []
        for idx in top_indices:
            result = SearchResult(
                index=int(idx),
                similarity=float(similarities[idx]),
                image_hash=self.image_hashes[idx],
                metadata=self.metadata[idx] if idx < len(self.metadata) else None
            )
            results.append(result)

        search_time = (time.time() - start_time) * 1000
        logger.info(f"Search completed in {search_time:.2f}ms. Found {len(results)} results")

        return results

    def batch_search(self,
                    query_embeddings: np.ndarray,
                    top_k: int = 5,
                    threshold: Optional[float] = None) -> List[List[SearchResult]]:
        """
        Search for multiple queries in batch.

        Args:
            query_embeddings: Array of query embeddings (M, D)
            top_k: Number of results per query
            threshold: Minimum similarity threshold

        Returns:
            List of result lists, one per query
        """
        results = []
        for query in query_embeddings:
            results.append(self.search(query, top_k, threshold))
        return results

    def find_duplicates(self,
                       threshold: float = 0.95) -> List[Dict]:
        """
        Find potential duplicate images based on embedding similarity.

        Args:
            threshold: Similarity threshold for duplicates

        Returns:
            List of dictionaries with image1_hash, image2_hash, and similarity
        """
        if self.embeddings is None or len(self.embeddings) < 2:
            return []

        duplicates = []
        n_embeddings = len(self.embeddings)

        # Compute pairwise similarities
        for i in range(n_embeddings):
            for j in range(i + 1, n_embeddings):
                similarity = self._compute_similarity(
                    self.embeddings[i], self.embeddings[j]
                )

                if similarity >= threshold:
                    duplicates.append({
                        'image1_hash': self.image_hashes[i],
                        'image2_hash': self.image_hashes[j],
                        'similarity': float(similarity)
                    })

        logger.info(f"Found {len(duplicates)} potential duplicates")
        return duplicates

    def remove_embedding(self, image_hash: str) -> bool:
        """
        Remove an embedding from the index.

        Args:
            image_hash: Hash of image to remove

        Returns:
            True if removed, False if not found
        """
        if image_hash not in self.image_hashes:
            return False

        idx = self.image_hashes.index(image_hash)

        # Remove from all storage
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        self.image_hashes.pop(idx)
        self.metadata.pop(idx)

        self.index_built = False  # Need to rebuild index
        logger.info(f"Removed embedding for {image_hash}")
        return True

    def find_anomalies(self, k: int = 5, top_n: int = 8) -> List[Tuple[str, float]]:
        """
        Find the most isolated images based on k-NN distances.

        Anomalies are images with highest mean distance to their k nearest neighbors.

        Args:
            k: Number of nearest neighbors to consider
            top_n: Number of top anomalies to return

        Returns:
            List of (image_hash, anomaly_score) tuples
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings available for anomaly detection")
            return []

        n_samples = len(self.embeddings)
        if n_samples <= k:
            logger.warning(f"Not enough samples ({n_samples}) for k={k} anomaly detection")
            k = max(1, n_samples - 1)

        anomaly_scores = []

        # For each image, compute mean distance to k-NN
        for idx in range(n_samples):
            # Compute distances to all other images
            distances = []
            for other_idx in range(n_samples):
                if idx != other_idx:
                    # Use cosine distance (1 - cosine similarity)
                    similarity = self._compute_similarity(
                        self.embeddings[idx],
                        self.embeddings[other_idx]
                    )
                    distance = 1.0 - similarity
                    distances.append(distance)

            if not distances:
                continue

            # Sort distances and take mean of k nearest
            distances.sort()
            k_nearest = min(k, len(distances))
            k_nearest_distances = distances[:k_nearest]
            mean_distance = np.mean(k_nearest_distances) if k_nearest_distances else 0

            # Store with image hash
            image_hash = self.image_hashes[idx]
            anomaly_scores.append((image_hash, mean_distance))

        # Sort by anomaly score (highest = most anomalous)
        anomaly_scores.sort(key=lambda x: x[1], reverse=True)

        # Log results
        logger.info(f"Found {len(anomaly_scores)} anomaly scores, returning top {top_n}")

        return anomaly_scores[:top_n]

    def save_index(self, output_path: Union[str, Path]) -> None:
        """
        Save the search index to disk.

        Args:
            output_path: Path to save index
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self.embeddings is not None:
            np.save(output_path.with_suffix('.npy'), self.embeddings)

        # Save metadata
        index_data = {
            'image_hashes': self.image_hashes,
            'metadata': self.metadata,
            'distance_metric': self.distance_metric,
            'n_embeddings': len(self.embeddings) if self.embeddings is not None else 0,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0
        }

        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved index to {output_path}")

    def load_index(self, index_path: Union[str, Path]) -> None:
        """
        Load search index from disk.

        Args:
            index_path: Path to index files
        """
        index_path = Path(index_path)

        # Load embeddings
        embeddings_path = index_path.with_suffix('.npy')
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)

        # Load metadata
        metadata_path = index_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                index_data = json.load(f)

            self.image_hashes = index_data['image_hashes']
            self.metadata = index_data['metadata']
            self.distance_metric = index_data.get('distance_metric', 'cosine')

            logger.info(f"Loaded index with {len(self.image_hashes)} embeddings")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")

        self.index_built = False

    def get_statistics(self) -> Dict:
        """
        Get statistics about the search index.

        Returns:
            Dictionary of statistics
        """
        if self.embeddings is None:
            return {
                'n_embeddings': 0,
                'embedding_dim': 0,
                'index_built': False,
                'distance_metric': self.distance_metric
            }

        # Compute similarity distribution
        similarities = []
        if len(self.embeddings) > 1:
            # Sample pairwise similarities
            n_samples = min(100, len(self.embeddings))
            indices = np.random.choice(len(self.embeddings), n_samples, replace=False)

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = self._compute_similarity(
                        self.embeddings[indices[i]],
                        self.embeddings[indices[j]]
                    )
                    similarities.append(float(sim))

        stats = {
            'n_embeddings': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'index_built': self.index_built,
            'distance_metric': self.distance_metric,
            'memory_mb': self.embeddings.nbytes / (1024 * 1024)
        }

        if similarities:
            stats['similarity_distribution'] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'median': np.median(similarities)
            }

        return stats

    # Private helper methods
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize a single embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize multiple embeddings to unit length."""
        if len(embeddings.shape) == 1:
            return self._normalize_embedding(embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _cosine_similarity(self,
                          query: np.ndarray,
                          embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and embeddings."""
        if len(embeddings.shape) == 1:
            # Single embedding
            return np.dot(query, embeddings)

        # Multiple embeddings
        return np.dot(embeddings, query)

    def _euclidean_distance(self,
                          query: np.ndarray,
                          embeddings: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance between query and embeddings."""
        if len(embeddings.shape) == 1:
            # Single embedding
            return np.linalg.norm(query - embeddings)

        # Multiple embeddings
        return np.linalg.norm(embeddings - query, axis=1)

    def _compute_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings."""
        if self.distance_metric == "cosine":
            embedding1 = self._normalize_embedding(embedding1)
            embedding2 = self._normalize_embedding(embedding2)
            return float(np.dot(embedding1, embedding2))
        elif self.distance_metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert to similarity (0-1 range)
            return float(1.0 / (1.0 + distance))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")


def create_similarity_search(config: Optional[Config] = None,
                           distance_metric: str = "cosine") -> SimilaritySearch:
    """
    Factory function to create a similarity search engine.

    Args:
        config: Configuration object
        distance_metric: Distance metric to use

    Returns:
        SimilaritySearch instance
    """
    return SimilaritySearch(config, distance_metric)


if __name__ == "__main__":
    # Example usage
    config = Config()
    search_engine = create_similarity_search(config)

    # Create some dummy embeddings for testing
    n_samples = 100
    embedding_dim = 2048
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    hashes = [f"image_{i:04d}" for i in range(n_samples)]

    # Add to index
    search_engine.add_embeddings(embeddings, hashes)
    search_engine.build_index()

    # Search for similar images
    query = embeddings[0]
    results = search_engine.search(query, top_k=5)

    for result in results:
        print(f"Hash: {result.image_hash}, Similarity: {result.similarity:.4f}")
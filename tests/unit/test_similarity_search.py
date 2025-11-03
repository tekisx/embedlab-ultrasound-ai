"""Tests for similarity search and anomaly detection."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from src.core.similarity_search import SimilaritySearch, SearchResult


class TestSimilaritySearch:
    """Test suite for similarity search functionality."""

    def test_search_initialization(self):
        """Test similarity search initialization."""
        searcher = SimilaritySearch(distance_metric='cosine')
        assert searcher.distance_metric == 'cosine'
        assert searcher.embeddings is None
        assert not searcher.index_built

    def test_add_and_search_embeddings(self):
        """Test adding embeddings and searching."""
        searcher = SimilaritySearch()

        # Create test embeddings
        embeddings = np.random.randn(5, 128)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        hashes = [f"hash_{i}" for i in range(5)]

        # Add embeddings
        searcher.add_embeddings(embeddings, hashes)

        # Build index
        searcher.build_index()
        assert searcher.index_built

        # Search with query
        query = embeddings[0]
        results = searcher.search(query, top_k=3)

        assert len(results) == 3
        assert results[0].image_hash == "hash_0"
        assert results[0].similarity >= 0.99  # Should match itself

    def test_find_duplicates(self):
        """Test duplicate detection."""
        searcher = SimilaritySearch()

        # Create embeddings with duplicates
        base_emb = np.random.randn(128)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Add original
        searcher.add_embeddings(np.array([base_emb]), ["original"])

        # Add near duplicate (add small noise)
        duplicate = base_emb + np.random.randn(128) * 0.01
        duplicate = duplicate / np.linalg.norm(duplicate)
        searcher.add_embeddings(np.array([duplicate]), ["duplicate"])

        # Add different embedding
        different = np.random.randn(128)
        different = different / np.linalg.norm(different)
        searcher.add_embeddings(np.array([different]), ["different"])

        # Find duplicates
        duplicates = searcher.find_duplicates(threshold=0.95)

        assert len(duplicates) > 0
        found_pair = False
        for dup in duplicates:
            if (dup['image1_hash'] == "original" and dup['image2_hash'] == "duplicate") or \
               (dup['image1_hash'] == "duplicate" and dup['image2_hash'] == "original"):
                found_pair = True
                break
        assert found_pair

    def test_find_anomalies(self):
        """Test k-NN based anomaly detection."""
        searcher = SimilaritySearch()

        # Create normal cluster of embeddings that are similar to each other
        base_normal = np.ones(128) * 0.5
        normal_embeddings = []
        normal_hashes = []
        for i in range(10):
            # Add small noise to base vector
            emb = base_normal + np.random.randn(128) * 0.01
            emb = emb / np.linalg.norm(emb)
            normal_embeddings.append(emb)
            normal_hashes.append(f"normal_{i}")

        searcher.add_embeddings(np.array(normal_embeddings), normal_hashes)

        # Add outlier embeddings that are very different from the normal cluster
        # Outlier 1: Orthogonal to the normal cluster
        outlier1 = np.zeros(128)
        outlier1[0] = 1.0  # First dimension only
        searcher.add_embeddings(np.array([outlier1]), ["outlier_1"])

        # Outlier 2: Opposite to the normal cluster
        outlier2 = -base_normal
        outlier2 = outlier2 / np.linalg.norm(outlier2)
        searcher.add_embeddings(np.array([outlier2]), ["outlier_2"])

        # Find anomalies
        anomalies = searcher.find_anomalies(k=3, top_n=3)

        assert len(anomalies) == 3
        # At least one outlier should be in top 2 anomalies
        anomaly_hashes = [hash for hash, score in anomalies]
        assert "outlier_1" in anomaly_hashes or "outlier_2" in anomaly_hashes

        # Anomaly scores should be sorted (highest first)
        scores = [score for hash, score in anomalies]
        assert scores == sorted(scores, reverse=True)

    def test_find_anomalies_empty(self):
        """Test anomaly detection with no embeddings."""
        searcher = SimilaritySearch()
        anomalies = searcher.find_anomalies(k=5, top_n=8)
        assert anomalies == []

    def test_find_anomalies_few_samples(self):
        """Test anomaly detection with fewer samples than k."""
        searcher = SimilaritySearch()

        # Add only 3 embeddings
        embeddings = []
        hashes = []
        for i in range(3):
            emb = np.random.randn(128)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            hashes.append(f"sample_{i}")

        searcher.add_embeddings(np.array(embeddings), hashes)

        # Request k=5 but only have 3 samples
        anomalies = searcher.find_anomalies(k=5, top_n=2)

        # Should still work with adjusted k
        assert len(anomalies) == 2
        assert all(score >= 0 for _, score in anomalies)

    def test_save_and_load_index(self):
        """Test saving and loading search index."""
        searcher = SimilaritySearch()

        # Add test data
        embeddings = []
        hashes = []
        for i in range(5):
            emb = np.random.randn(128)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            hashes.append(f"hash_{i}")

        searcher.add_embeddings(np.array(embeddings), hashes)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index"

            # Save index
            searcher.save_index(index_path)

            # Load in new searcher
            new_searcher = SimilaritySearch()
            new_searcher.load_index(index_path)

            assert len(new_searcher.image_hashes) == 5
            assert new_searcher.embeddings.shape == (5, 128)

    def test_compute_similarity(self):
        """Test similarity computation between embeddings."""
        searcher = SimilaritySearch()

        # Test identical embeddings
        emb1 = np.array([1, 0, 0])
        emb1 = emb1 / np.linalg.norm(emb1)

        similarity = searcher._compute_similarity(emb1, emb1)
        assert pytest.approx(similarity, 0.001) == 1.0

        # Test orthogonal embeddings
        emb2 = np.array([0, 1, 0])
        emb2 = emb2 / np.linalg.norm(emb2)

        similarity = searcher._compute_similarity(emb1, emb2)
        assert pytest.approx(similarity, 0.001) == 0.0

        # Test opposite embeddings
        emb3 = -emb1
        similarity = searcher._compute_similarity(emb1, emb3)
        assert pytest.approx(similarity, 0.001) == -1.0

    def test_anomaly_detection_consistency(self):
        """Test that anomaly detection gives consistent results."""
        np.random.seed(42)
        searcher = SimilaritySearch()

        # Create dataset with known outliers
        embeddings = []
        hashes = []
        for i in range(20):
            if i < 18:
                # Normal samples
                emb = np.random.randn(128) * 0.1 + np.ones(128) * 0.5
            else:
                # Outliers
                emb = np.random.randn(128) * 2.0
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            hashes.append(f"sample_{i}")

        searcher.add_embeddings(np.array(embeddings), hashes)

        # Run anomaly detection multiple times
        results = []
        for _ in range(3):
            anomalies = searcher.find_anomalies(k=5, top_n=5)
            results.append([h for h, s in anomalies])

        # Results should be identical
        assert all(r == results[0] for r in results)

    def test_invalid_distance_metric(self):
        """Test initialization with invalid distance metric."""
        with pytest.raises(ValueError):
            searcher = SimilaritySearch(distance_metric='invalid')

    def test_search_with_threshold(self):
        """Test search with similarity threshold."""
        searcher = SimilaritySearch()

        # Add embeddings
        embeddings = []
        hashes = []
        for i in range(5):
            emb = np.random.randn(128)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            hashes.append(f"hash_{i}")

        searcher.add_embeddings(np.array(embeddings), hashes)

        searcher.build_index()

        # Search with high threshold
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)

        results = searcher.search(query, top_k=10, threshold=0.99)

        # With random embeddings and high threshold, should get few/no results
        assert len(results) <= 1  # At most the query itself if it's in the index
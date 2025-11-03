#!/usr/bin/env python3
"""
EmbedLab - Image Embedding Laboratory CLI

Matches the exact interface specified in the transcription.
"""

import click
import sys
from pathlib import Path
import json
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.utils.logging import get_logger
from src.core.index_manager import IndexManager
from src.core.similarity_search import SimilaritySearch

logger = get_logger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """EmbedLab - Image embedding and similarity search tool."""
    # Don't show banner to match transcription expectations
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('--images-dir', required=True, type=click.Path(exists=True),
              help='Directory containing images to embed')
@click.option('--out', required=True, type=click.Path(),
              help='Output directory for index')
@click.pass_context
def embed(ctx, images_dir, out):
    """Compute embeddings for every image and persist to index."""
    config = Config()
    index_path = Path(out)
    manager = IndexManager(config, index_path)

    images_dir = Path(images_dir)

    # Find all images
    all_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        all_paths.extend(images_dir.glob(f"**/{ext}"))

    # Log stats as required
    start_time = time.time()
    print(f"Found {len(all_paths)} images to process")
    print(f"Using backbone: ResNet50")
    print(f"Embedding dimension: 2048")

    # Process images
    results = manager.add_batch(
        all_paths,
        batch_size=32,
        generate_embeddings=True
    )

    # Save index
    manager.save_index()

    elapsed = time.time() - start_time
    print(f"Processing complete: {results['successful']}/{results['total']} images")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Index saved to: {index_path}")


@cli.command()
@click.option('--index', required=True, type=click.Path(exists=True),
              help='Path to index directory')
@click.option('--query-dir', required=True, type=click.Path(exists=True),
              help='Directory containing query images')
@click.option('--k', type=int, default=5,
              help='Number of results to return')
@click.option('--json', 'output_json', is_flag=True,
              help='Output as JSON')
@click.pass_context
def search(ctx, index, query_dir, k, output_json):
    """For each query image, return top-k most similar images."""
    config = Config()
    index_path = Path(index)
    manager = IndexManager(config, index_path)

    # Load existing index
    manager.load_index()

    query_dir = Path(query_dir)

    # Find all query images
    query_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        query_paths.extend(query_dir.glob(f"**/{ext}"))

    # Process each query
    for query_path in query_paths:
        # Search for similar images
        results = manager.search(str(query_path), top_k=k)

        # Format output as specified in transcription
        if output_json:
            output = {
                "query": str(query_path).replace('\\', '/'),
                "results": []
            }

            for result in results:
                # Get original path from registry
                original_path = manager.get_image_path(result.get('image_hash'))
                if original_path:
                    output["results"].append({
                        "path": str(original_path).replace('\\', '/'),
                        "score": round(result['similarity'], 2)
                    })

            # Print JSON for each query
            print(json.dumps(output))
        else:
            print(f"\nQuery: {query_path}")
            for i, result in enumerate(results, 1):
                original_path = manager.get_image_path(result.get('image_hash'))
                print(f"  {i}. {original_path} (score: {result['similarity']:.4f})")


@cli.command()
@click.option('--index', required=True, type=click.Path(exists=True),
              help='Path to index directory')
@click.option('--dup-threshold', type=float, default=0.92,
              help='Similarity threshold for duplicates')
@click.option('--anomaly-top', type=int, default=8,
              help='Number of top anomalies to return')
@click.option('--json', 'output_json', is_flag=True,
              help='Output as JSON')
@click.pass_context
def analyze(ctx, index, dup_threshold, anomaly_top, output_json):
    """Detect duplicates and anomalies."""
    config = Config()
    index_path = Path(index)
    manager = IndexManager(config, index_path)

    # Load existing index
    manager.load_index()

    # Find duplicates
    duplicate_pairs = manager.find_duplicates(threshold=dup_threshold)

    # Convert pairs to groups
    duplicate_groups = []
    if duplicate_pairs:
        groups = {}
        for pair in duplicate_pairs:
            hash1, hash2 = pair['image1_hash'], pair['image2_hash']
            found_group = None

            for group_id, group_hashes in groups.items():
                if hash1 in group_hashes or hash2 in group_hashes:
                    found_group = group_id
                    break

            if found_group:
                groups[found_group].add(hash1)
                groups[found_group].add(hash2)
            else:
                groups[len(groups)] = {hash1, hash2}

        # Convert to paths
        for group in groups.values():
            group_paths = []
            for hash_val in group:
                path = manager.get_image_path(hash_val)
                if path:
                    group_paths.append(str(path).replace('\\', '/'))
            if len(group_paths) >= 2:
                duplicate_groups.append(group_paths)

    # Find anomalies
    anomaly_list = manager.similarity_search.find_anomalies(k=5, top_n=anomaly_top)

    # Convert to paths
    anomaly_paths = []
    for hash_val, score in anomaly_list:
        path = manager.get_image_path(hash_val)
        if path:
            anomaly_paths.append(str(path).replace('\\', '/'))

    # Output
    if output_json:
        output = {
            "duplicate_groups": duplicate_groups,
            "anomalies": anomaly_paths
        }
        print(json.dumps(output))
    else:
        print(f"Found {len(duplicate_groups)} duplicate groups")
        for i, group in enumerate(duplicate_groups, 1):
            print(f"  Group {i}: {group}")

        print(f"\nTop {len(anomaly_paths)} anomalies:")
        for i, path in enumerate(anomaly_paths, 1):
            print(f"  {i}. {path}")


if __name__ == "__main__":
    cli()
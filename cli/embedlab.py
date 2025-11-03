#!/usr/bin/env python3
"""
EmbedLab - Medical Image Embedding Laboratory CLI

A production-ready CLI for medical ultrasound image embeddings with
PHI-safe logging and comprehensive functionality.
"""

import click
import sys
from pathlib import Path
import json
import time
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.utils.logging import get_logger
from src.core.index_manager import IndexManager

# Initialize rich console for beautiful output
console = Console()
logger = get_logger(__name__)

# ASCII art banner (Windows-safe)
BANNER = """
================================================================

    EMBEDLAB - Medical Image Embedding Laboratory
    ------------------------------------------------

    Version:     1.0.0
    Features:    PHI-Safe, Production-Ready
    Specialty:   Ultrasound AI

================================================================
"""


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--index-path', '-i', type=click.Path(),
              default='index', help='Path to index directory')
@click.pass_context
def cli(ctx, config, index_path):
    """
    EmbedLab - Medical Image Embedding Laboratory

    A comprehensive CLI for managing medical ultrasound image embeddings
    with PHI-safe operations and production-ready features.
    """
    # Show banner
    console.print(BANNER, style="bold cyan")

    # Initialize configuration
    if config:
        ctx.obj = Config.load(config)
        console.print(f"[green]Loaded configuration from {config}[/green]")
    else:
        ctx.obj = Config()

    # Store index path in context
    ctx.obj.index_path = Path(index_path)


@cli.command()
@click.argument('image_paths', nargs=-1, required=True,
                type=click.Path(exists=True))
@click.option('--batch-size', '-b', default=32,
              help='Batch size for processing')
@click.option('--no-embedding', is_flag=True,
              help='Skip embedding generation')
@click.pass_obj
def add(config, image_paths, batch_size, no_embedding):
    """Add images to the embedding index."""
    manager = IndexManager(config, config.index_path)

    # Convert to Path objects
    paths = [Path(p) for p in image_paths]

    # Expand directories
    all_paths = []
    for path in paths:
        if path.is_dir():
            # Find all images in directory
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                all_paths.extend(path.glob(f"**/{ext}"))
        else:
            all_paths.append(path)

    console.print(f"\n[cyan]Found {len(all_paths)} images to process[/cyan]")

    # Process images with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing images...", total=len(all_paths))

        results = manager.add_batch(
            all_paths,
            batch_size=batch_size,
            generate_embeddings=not no_embedding
        )

        progress.update(task, completed=len(all_paths))

    # Display results
    table = Table(title="Processing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Images", str(results['total']))
    table.add_row("Successful", str(results['successful']))
    table.add_row("Failed", str(results['failed']))
    table.add_row("Skipped (Duplicates)", str(results['skipped']))

    console.print(table)

    # Save index
    manager.save_index()
    console.print(f"\n[green]✓ Index saved to {config.index_path}[/green]")


@cli.command()
@click.argument('query_image', type=click.Path(exists=True))
@click.option('--top-k', '-k', default=5,
              help='Number of results to return')
@click.option('--threshold', '-t', default=0.7,
              help='Similarity threshold (0-1)')
@click.pass_obj
def search(config, query_image, top_k, threshold):
    """Search for similar images in the index."""
    manager = IndexManager(config, config.index_path)

    # Load existing index
    manager.load_index()

    console.print(f"\n[cyan]Searching for images similar to: {query_image}[/cyan]")

    # Perform search
    with console.status("Searching...", spinner="dots"):
        results = manager.search(query_image, top_k=top_k, threshold=threshold)

    if not results:
        console.print("[red]No similar images found[/red]")
        return

    # Display results
    table = Table(title=f"Top {len(results)} Similar Images")
    table.add_column("#", style="cyan")
    table.add_column("Image Hash", style="yellow")
    table.add_column("Similarity", style="green")
    table.add_column("Quality", style="blue")

    for idx, result in enumerate(results, 1):
        quality = result.get('image_info', {}).get('quality_score', 'N/A')
        if isinstance(quality, float):
            quality = f"{quality:.3f}"

        table.add_row(
            str(idx),
            result['image_hash'][:16] + "...",
            f"{result['similarity']:.4f}",
            quality
        )

    console.print(table)


@cli.command()
@click.option('--threshold', '-t', default=0.95,
              help='Similarity threshold for duplicates')
@click.pass_obj
def duplicates(config, threshold):
    """Find duplicate images in the index."""
    manager = IndexManager(config, config.index_path)

    # Load existing index
    manager.load_index()

    console.print(f"\n[cyan]Finding duplicates with threshold {threshold}[/cyan]")

    # Find duplicates
    with console.status("Analyzing...", spinner="dots"):
        duplicates = manager.find_duplicates(threshold=threshold)

    if not duplicates:
        console.print("[green]No duplicates found[/green]")
        return

    # Display duplicates
    table = Table(title=f"Found {len(duplicates)} Duplicate Pairs")
    table.add_column("Image 1", style="yellow")
    table.add_column("Image 2", style="yellow")
    table.add_column("Similarity", style="red")

    for dup in duplicates[:10]:  # Show first 10
        table.add_row(
            dup['image1_hash'][:16] + "...",
            dup['image2_hash'][:16] + "...",
            f"{dup['similarity']:.4f}"
        )

    console.print(table)

    if len(duplicates) > 10:
        console.print(f"\n[dim]... and {len(duplicates) - 10} more pairs[/dim]")


@cli.command()
@click.pass_obj
def stats(config):
    """Display index statistics."""
    manager = IndexManager(config, config.index_path)

    # Load existing index
    manager.load_index()

    # Get statistics
    stats = manager.get_statistics()

    # Create statistics panel
    stats_text = Text()
    stats_text.append("Index Statistics\n\n", style="bold cyan")
    stats_text.append(f"Total Images: ", style="white")
    stats_text.append(f"{stats['n_images']}\n", style="green")
    stats_text.append(f"Total Embeddings: ", style="white")
    stats_text.append(f"{stats['n_embeddings']}\n", style="green")
    stats_text.append(f"Index Location: ", style="white")
    stats_text.append(f"{stats['index_path']}\n", style="blue")

    if 'quality_distribution' in stats:
        qd = stats['quality_distribution']
        stats_text.append(f"\nQuality Distribution:\n", style="bold white")
        stats_text.append(f"  Mean: {qd['mean']:.3f}\n", style="green")
        stats_text.append(f"  Std:  {qd['std']:.3f}\n", style="green")
        stats_text.append(f"  Range: {qd['min']:.3f} - {qd['max']:.3f}\n", style="green")

    if 'similarity_search' in stats:
        ss = stats['similarity_search']
        stats_text.append(f"\nSearch Index:\n", style="bold white")
        stats_text.append(f"  Memory: {ss.get('memory_mb', 0):.2f} MB\n", style="yellow")
        stats_text.append(f"  Metric: {ss.get('distance_metric', 'cosine')}\n", style="yellow")

    panel = Panel(stats_text, title="EmbedLab Statistics", border_style="cyan")
    console.print(panel)


@cli.command()
@click.pass_obj
def validate(config):
    """Validate index integrity."""
    manager = IndexManager(config, config.index_path)

    # Load existing index
    manager.load_index()

    console.print("\n[cyan]Validating index integrity...[/cyan]")

    # Validate
    with console.status("Checking...", spinner="dots"):
        validation = manager.validate_index()

    if validation['valid']:
        console.print("[green]✓ Index validation passed![/green]")
    else:
        console.print("[red]✗ Index validation failed![/red]")
        console.print("\nIssues found:")
        for issue in validation['issues']:
            console.print(f"  • {issue}", style="yellow")

    # Show counts
    console.print(f"\nImages in registry: {validation['n_images']}")
    console.print(f"Embeddings in index: {validation['n_embeddings']}")


@cli.command()
@click.argument('output_path', type=click.Path())
@click.pass_obj
def export(config, output_path):
    """Export index to a different location."""
    manager = IndexManager(config, config.index_path)

    # Load existing index
    manager.load_index()

    console.print(f"\n[cyan]Exporting index to {output_path}[/cyan]")

    # Export with progress
    with console.status("Exporting...", spinner="dots"):
        manager.save_index(output_path)

    console.print(f"[green]✓ Index exported successfully[/green]")


@cli.command()
@click.argument('import_path', type=click.Path(exists=True))
@click.pass_obj
def import_index(config, import_path):
    """Import index from another location."""
    manager = IndexManager(config, config.index_path)

    console.print(f"\n[cyan]Importing index from {import_path}[/cyan]")

    # Import with progress
    with console.status("Importing...", spinner="dots"):
        manager.load_index(import_path)

    # Get statistics
    stats = manager.get_statistics()

    console.print(f"[green]✓ Imported {stats['n_images']} images successfully[/green]")


@cli.command()
@click.pass_obj
def benchmark(config):
    """Run performance benchmarks."""
    manager = IndexManager(config, config.index_path)

    console.print("\n[cyan]Running performance benchmarks...[/cyan]")

    # Find test images
    test_path = Path("assets/images/breast_ultrasound/Dataset_BUSI_with_GT/benign")
    if not test_path.exists():
        console.print("[red]Test images not found. Please download the dataset first.[/red]")
        return

    test_images = list(test_path.glob("*.png"))[:10]

    if len(test_images) < 2:
        console.print("[red]Not enough test images for benchmarking[/red]")
        return

    results = {}

    # Benchmark image loading
    with console.status("Benchmarking image loading...", spinner="dots"):
        start = time.time()
        for img in test_images:
            _ = manager.image_processor.load_image(img)
        results['image_loading'] = (time.time() - start) / len(test_images) * 1000

    # Benchmark embedding generation
    if manager.embedding_engine:
        with console.status("Benchmarking embedding generation...", spinner="dots"):
            start = time.time()
            for img in test_images[:3]:  # Only test a few
                _ = manager.embedding_engine.generate_embedding(img)
            results['embedding_gen'] = (time.time() - start) / 3 * 1000

    # Display results
    table = Table(title="Performance Benchmarks")
    table.add_column("Operation", style="cyan")
    table.add_column("Time (ms)", style="green")

    table.add_row("Image Loading", f"{results.get('image_loading', 0):.2f}")
    table.add_row("Embedding Generation", f"{results.get('embedding_gen', 0):.2f}")

    console.print(table)


@cli.command()
def info():
    """Display system information."""
    import torch
    import platform

    info_text = Text()
    info_text.append("System Information\n\n", style="bold cyan")

    # Python info
    info_text.append("Python: ", style="white")
    info_text.append(f"{sys.version.split()[0]}\n", style="green")

    # Platform info
    info_text.append("Platform: ", style="white")
    info_text.append(f"{platform.platform()}\n", style="green")

    # PyTorch info
    info_text.append("PyTorch: ", style="white")
    info_text.append(f"{torch.__version__}\n", style="green")

    info_text.append("CUDA Available: ", style="white")
    info_text.append(f"{'Yes' if torch.cuda.is_available() else 'No'}\n",
                     style="green" if torch.cuda.is_available() else "red")

    # Dataset info
    dataset_path = Path("assets/images/breast_ultrasound")
    if dataset_path.exists():
        n_images = len(list(dataset_path.rglob("*.png")))
        info_text.append("\nDataset: ", style="white")
        info_text.append(f"Breast Ultrasound ({n_images} images)\n", style="green")
    else:
        info_text.append("\nDataset: ", style="white")
        info_text.append("Not downloaded\n", style="red")

    panel = Panel(info_text, title="System Information", border_style="cyan")
    console.print(panel)


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.error(f"CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# Plan to Complete Missing Core Features

## Issue Analysis

We have **2 critical issues** that don't match the interview requirements:

1. **WRONG COMMAND NAMES** - We renamed commands when we should have kept exact names:
   - ❌ `add` → Should be `embed`
   - ✅ `search` → Correct
   - ❌ `duplicates` → Should be `analyze` (with both duplicates AND anomalies)

2. **MISSING ANOMALY DETECTION** - The `analyze` command must output BOTH:
   - ✅ `duplicate_groups` - We have this
   - ❌ `anomalies` - We're missing this

## Implementation Plan

### Step 1: Fix Command Names (~10 minutes)

**File:** `cli/embedlab.py`

1. Change `@cli.command('add')` to `@cli.command('embed')`
2. Change `@cli.command('duplicates')` to `@cli.command('analyze')`
3. Update all help text to match

### Step 2: Implement k-NN Anomaly Detection (~30 minutes)

**File:** `src/core/similarity_search.py`

Add new method to `SimilaritySearch` class:

```python
def find_anomalies(self, k: int = 5, top_n: int = 8) -> List[Tuple[str, float]]:
    """
    Find the most isolated images based on k-NN distances.

    Anomalies are images with highest mean distance to their k nearest neighbors.

    Args:
        k: Number of nearest neighbors to consider
        top_n: Number of top anomalies to return

    Returns:
        List of (image_path, anomaly_score) tuples
    """
    if self.embeddings is None or len(self.embeddings) == 0:
        return []

    anomaly_scores = []

    # For each image, compute mean distance to k-NN
    for idx in range(len(self.embeddings)):
        # Get distances to all other images
        distances = []
        for other_idx in range(len(self.embeddings)):
            if idx != other_idx:
                # Use 1 - similarity to get distance
                similarity = self._compute_similarity(
                    self.embeddings[idx],
                    self.embeddings[other_idx]
                )
                distance = 1.0 - similarity
                distances.append(distance)

        # Sort distances and take mean of k nearest
        distances.sort()
        k_nearest_distances = distances[:min(k, len(distances))]
        mean_distance = np.mean(k_nearest_distances) if k_nearest_distances else 0

        # Get image path from registry
        image_path = self.registry.get_image_path(idx)
        anomaly_scores.append((image_path, mean_distance))

    # Sort by anomaly score (highest = most anomalous)
    anomaly_scores.sort(key=lambda x: x[1], reverse=True)

    return anomaly_scores[:top_n]
```

### Step 3: Update Analyze Command (~20 minutes)

**File:** `cli/embedlab.py`

Merge duplicates command into analyze with both outputs:

```python
@cli.command('analyze')
@click.option('--index', '-i', 'index_path',
              type=click.Path(exists=True),
              help='Path to index directory')
@click.option('--dup-threshold', default=0.92,
              type=float,
              help='Similarity threshold for duplicates')
@click.option('--anomaly-top', default=8,
              type=int,
              help='Number of top anomalies to return')
@click.option('--k', default=5,
              type=int,
              help='Number of neighbors for anomaly detection')
@click.option('--json', 'output_json',
              is_flag=True,
              help='Output as JSON')
@click.pass_context
def analyze(ctx, index_path, dup_threshold, anomaly_top, k, output_json):
    """
    Analyze index for duplicates and anomalies.

    Outputs both duplicate groups and anomalous images.
    """
    # Load index
    index_mgr = ctx.obj['index_manager']
    if index_path:
        index_mgr.load_index(Path(index_path))

    # Find duplicates
    console.print("\n[yellow]Finding duplicates...[/yellow]")
    duplicates = index_mgr.find_duplicates(threshold=dup_threshold)

    # Find anomalies (NEW)
    console.print("\n[yellow]Finding anomalies...[/yellow]")
    anomalies = index_mgr.searcher.find_anomalies(k=k, top_n=anomaly_top)

    if output_json:
        # Format as per requirements
        output = {
            "duplicate_groups": duplicates,
            "anomalies": [path for path, score in anomalies]
        }
        console.print_json(data=output)
    else:
        # Pretty print results
        # ... existing duplicate printing ...

        # Add anomaly printing
        console.print(f"\n[red]Top {anomaly_top} Anomalies:[/red]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Image Path")
        table.add_column("Anomaly Score", justify="right")

        for i, (path, score) in enumerate(anomalies, 1):
            table.add_row(str(i), path, f"{score:.4f}")

        console.print(table)
```

### Step 4: Add Tests (~15 minutes)

**File:** `tests/unit/test_similarity_search.py` (NEW)

```python
def test_anomaly_detection():
    """Test k-NN based anomaly detection."""
    # Create embeddings with clear outliers
    normal_embeddings = np.random.randn(10, 128)
    outlier_embedding = np.random.randn(1, 128) * 10  # Very different

    all_embeddings = np.vstack([normal_embeddings, outlier_embedding])

    searcher = SimilaritySearch()
    searcher.build_index(all_embeddings, image_paths)

    anomalies = searcher.find_anomalies(k=3, top_n=3)

    # Outlier should be in top anomalies
    assert len(anomalies) == 3
    anomaly_paths = [path for path, _ in anomalies]
    assert 'outlier.png' in anomaly_paths[:2]  # Should be highly ranked
```

### Step 5: Update README (~5 minutes)

Change all references:
- `add` → `embed`
- `duplicates` → `analyze`
- Add description of anomaly detection

## Time Estimate

**Total: ~80 minutes**

1. Fix command names: 10 min
2. Implement k-NN anomaly: 30 min
3. Update analyze command: 20 min
4. Add tests: 15 min
5. Update documentation: 5 min

## Impact on Existing Code

- **CLI changes:** 2 commands renamed
- **New algorithm:** k-NN anomaly detection in similarity_search.py
- **Tests:** Need new test file for similarity search
- **Documentation:** README needs updates

## Why This Matters

**Without these fixes:**
- ❌ Commands don't match spec = immediate failure
- ❌ Missing anomaly detection = core requirement not met
- ❌ Would fail automated grading that expects exact command names

**With these fixes:**
- ✅ 100% core requirements met
- ✅ Commands match specification exactly
- ✅ Full `analyze` command with both outputs
- ✅ Ready for automated grading
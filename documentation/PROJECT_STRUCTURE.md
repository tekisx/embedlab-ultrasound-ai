# Project Structure - Image Embedding Lab

## Overview
This document defines the complete project structure optimized for isolated testing, medical imaging best practices, and professional Python packaging standards.

## Directory Structure

```
Interview Project/
├── src/                                    # Main source code
│   ├── __init__.py
│   ├── __main__.py                        # Allows: python -m src.cli.embedlab
│   ├── utils/                             # Utility modules (no dependencies)
│   │   ├── __init__.py
│   │   ├── logging.py                     # PHI-safe logging utilities
│   │   └── config.py                      # Seed management, configuration
│   ├── core/                              # Core functionality modules
│   │   ├── __init__.py
│   │   ├── image_processor.py             # Image loading, preprocessing, quality checks
│   │   ├── embedding_engine.py            # Model management, embedding computation
│   │   ├── index_manager.py               # Storage/retrieval with memory mapping
│   │   ├── similarity_search.py           # Cosine similarity, ranking
│   │   ├── duplicate_detector.py          # Union-Find duplicate grouping
│   │   └── anomaly_detector.py            # k-NN anomaly detection
│   └── cli/                               # Command-line interface
│       ├── __init__.py
│       └── embedlab.py                    # Main CLI with argparse
│
├── tests/                                 # Test suite
│   ├── __init__.py
│   ├── conftest.py                        # Pytest fixtures and shared test config
│   ├── unit/                              # Unit tests (isolated, fast)
│   │   ├── __init__.py
│   │   ├── test_logging.py                # Test PHI-safe logging
│   │   ├── test_config.py                 # Test seed management
│   │   ├── test_image_processor.py        # Test preprocessing pipeline
│   │   ├── test_embedding_engine.py       # Test embedding computation
│   │   ├── test_index_manager.py          # Test storage/retrieval
│   │   ├── test_similarity_search.py      # Test similarity algorithms
│   │   ├── test_duplicate_detector.py     # Test Union-Find clustering
│   │   └── test_anomaly_detector.py       # Test k-NN anomaly detection
│   ├── integration/                       # Integration tests (full pipeline)
│   │   ├── __init__.py
│   │   └── test_end_to_end.py             # Test complete CLI workflows
│   └── fixtures/                          # Test data and utilities
│       ├── __init__.py
│       └── generate_synthetic_images.py   # Minimal synthetic images for unit tests
│
├── assets/                                # Input data (tracked in git)
│   ├── images/                            # ~20-30 downloaded medical images
│   ├── queries/                           # 6 query images for search testing
│   ├── weights/                           # Optional pre-cached ResNet50 weights
│   └── synthetic/                         # Minimal synthetic test images
│
├── index/                                 # Generated embeddings (in .gitignore)
│   ├── embeddings.npy                     # Memory-mapped embedding vectors
│   ├── metadata.json                      # File paths, hashes, quality scores
│   └── manifest.json                      # Model info, versions, timestamps
│
├── output/                                # Generated metrics (in .gitignore)
│   ├── metrics.json                       # Observability: timing, throughput, counts
│   ├── search_results.json                # Optional saved search results
│   └── analysis_results.json              # Optional saved analysis results
│
├── docs/                                  # Additional documentation
│   └── (reserved for future documentation)
│
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup for CLI installation
├── pyproject.toml                         # Modern Python packaging + mypy config
├── .gitignore                             # Exclude generated files
├── README.md                              # Main documentation
├── PROMPTS.md                             # AI assistance documentation
├── IMPLEMENTATION_PLAN.md                 # Implementation strategy (this doc)
├── Image_Embedding_Lab_Full_Transcription.md  # Assignment specification
└── PROJECT_STRUCTURE.md                   # This file

```

## Module Dependencies (Bottom-Up Testing Order)

### Layer 1: Utilities (No Dependencies)
1. **utils/logging.py** - Independent logging utilities
2. **utils/config.py** - Configuration and seed management (depends on logging)

### Layer 2: Core Processing (Depends on Utilities)
3. **core/image_processor.py** - Image preprocessing (depends on logging, config)
4. **core/embedding_engine.py** - Embedding computation (depends on image_processor, config)
5. **core/index_manager.py** - Storage/retrieval (depends on logging)

### Layer 3: Algorithms (Depends on Core)
6. **core/similarity_search.py** - Similarity algorithms (depends on index_manager)
7. **core/duplicate_detector.py** - Duplicate detection (depends on similarity_search)
8. **core/anomaly_detector.py** - Anomaly detection (depends on similarity_search)

### Layer 4: CLI (Depends on All)
9. **cli/embedlab.py** - Main CLI orchestrator (depends on all core modules)

## Testing Strategy

### Unit Tests (tests/unit/)
- **Purpose**: Test individual functions/classes in isolation
- **Dependencies**: Use mocks/fixtures for external dependencies
- **Speed**: Fast (<1s per test file)
- **Example**: Test image preprocessing without loading actual model

### Integration Tests (tests/integration/)
- **Purpose**: Test complete workflows end-to-end
- **Dependencies**: Use real modules and downloaded medical images
- **Speed**: Slower (several seconds)
- **Example**: Test full embed → search → analyze pipeline

### Test Fixtures (tests/fixtures/)
- **Purpose**: Provide synthetic data for unit tests
- **Contents**: Programmatically generated test images (blank, corrupted, edge cases)
- **Usage**: Import in conftest.py as pytest fixtures

## Image Acquisition Strategy

### Downloaded Real Images (assets/images/)
**Purpose**: Realistic testing with actual medical imaging edge cases

**Target Collection (~20-30 images):**
- 5-10 normal ultrasound images (baseline)
- 2-3 low contrast/dark images (exposure issues)
- 2-3 bright/overexposed images
- 3-5 near-duplicate pairs (consecutive frames)
- 2-3 different modalities (fetal, cardiac, abdominal)
- 2-3 various aspect ratios/resolutions
- 2-3 color-mapped ultrasound images
- 2-3 images with medical text overlays

**Acquisition Method:**
1. Use WebSearch to find public ultrasound datasets
2. Use Firecrawl to scrape dataset pages for image URLs
3. Use curl/wget to download images
4. Validate images programmatically (format, loadable, size)

**Public Dataset Sources to Search:**
- Kaggle ultrasound datasets
- NIH medical image repositories
- Academic research datasets (with public licenses)
- TCIA (The Cancer Imaging Archive) - has some ultrasound
- Zenodo medical imaging collections

### Synthetic Images (assets/synthetic/)
**Purpose**: Unit testing specific edge cases

**Minimal Set (generated programmatically):**
- Blank/all-black images (1x1, 10x10, 224x224)
- Corrupted pixel data
- Single-pixel images
- Extreme dimensions (1x10000, 10000x1)

## Type Hints Strategy

### Complete Type Coverage
Every function must have:
- Argument types
- Return types
- Optional/Union types where applicable

### Example:
```python
from typing import List, Dict, Tuple, Optional
import numpy as np

def compute_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray
) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb1: First embedding vector (1D numpy array)
        emb2: Second embedding vector (1D numpy array)

    Returns:
        Cosine similarity score in range [0, 1]

    Raises:
        ValueError: If embeddings have different dimensions
    """
    pass
```

### TypedDict for JSON Structures
```python
from typing import TypedDict, List

class SearchResult(TypedDict):
    query: str
    results: List[Dict[str, float]]
```

## CLI Entry Points

### setup.py Configuration
```python
setup(
    name='embedlab',
    entry_points={
        'console_scripts': [
            'embedlab=src.cli.embedlab:main',
        ],
    },
)
```

### Usage After Installation
```bash
# Install package
pip install -e .

# Run CLI
embedlab embed --images-dir ./assets/images --out ./index
embedlab search --index ./index --query-dir ./assets/queries --k 5 --json
embedlab analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json
```

## Professional Packaging

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "embedlab"
version = "0.1.0"
description = "Medical image embedding and similarity search tool"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy>=1.24",
    "Pillow>=10.0",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy>=1.0",
    "black>=23.0",
    "ruff>=0.1",
]

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

## .gitignore
```gitignore
# Generated directories
index/
output/
__pycache__/
.pytest_cache/
.mypy_cache/
*.egg-info/
dist/
build/

# Downloaded model weights
assets/weights/*.pth
assets/weights/*.pt

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## PHI Compliance Architecture

### Key Principles
1. **Never log actual filenames** - Use hashes or sequential IDs
2. **Sanitize all outputs** - Remove potential patient identifiers
3. **Audit trail** - Log access patterns without exposing data
4. **Configurable anonymization** - Allow different privacy levels

### Example Implementation
```python
import hashlib
from pathlib import Path

def phi_safe_identifier(file_path: Path) -> str:
    """Generate PHI-safe identifier for an image.

    Args:
        file_path: Path to image file

    Returns:
        SHA-256 hash of file path as identifier
    """
    return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
```

## Incremental Build Strategy

### Phase 1: Foundation
1. Create directory structure
2. Build utils/logging.py → test
3. Build utils/config.py → test

### Phase 2: Core Processing
4. Download medical images
5. Build core/image_processor.py → test with real images
6. Build core/embedding_engine.py → test determinism

### Phase 3: Storage & Retrieval
7. Build core/index_manager.py → test with numpy arrays
8. Verify memory mapping works

### Phase 4: Algorithms
9. Build core/similarity_search.py → test with known pairs
10. Build core/duplicate_detector.py → test with duplicates
11. Build core/anomaly_detector.py → test with outliers

### Phase 5: CLI Integration
12. Build cli/embedlab.py → integration tests
13. Test all three commands end-to-end

### Phase 6: Polish
14. Add documentation
15. Run mypy type checking
16. Performance profiling
17. Final testing

## Success Criteria

- ✓ All modules independently testable
- ✓ Complete type hint coverage (mypy --strict passes)
- ✓ Real medical images in test suite
- ✓ PHI-safe logging throughout
- ✓ Professional packaging (installable via pip)
- ✓ Comprehensive documentation
- ✓ All CLI commands working with proper help messages
- ✓ Performance metrics and observability

This structure ensures production-ready code that demonstrates both technical excellence and medical domain awareness.

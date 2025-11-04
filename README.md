# EmbedLab - Medical Image Embedding System

Interview project for Ultrasound AI. Built in ~2 hours. 68 tests passing.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Optional: Get real ultrasound data from Kaggle
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p assets/images/breast_ultrasound --unzip
```

## Commands

```bash
# Generate embeddings for images
python embedlab.py embed --images-dir ./assets/images --out ./index

# Search for similar images
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json

# Find duplicates and anomalies
python embedlab.py analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json
```

## What This Does

EmbedLab generates embeddings from medical images using ResNet50, finds similar images with cosine similarity, detects duplicates, and identifies anomalous images through k-NN analysis. All file paths are hashed before logging to ensure PHI safety.

## Implementation

### Core Requirements
- ✅ **embed**: Computes embeddings, persists to index
- ✅ **search**: Returns top-k similar images with cosine similarity
- ✅ **analyze**: Detects duplicate groups AND anomalies
- ✅ **ResNet50**: 2048-dimensional embeddings
- ✅ **Deterministic**: Fixed seeds for reproducible results
- ✅ **JSON output**: Structured output with paths and scores

### Additional Features Implemented
- Batch processing with configurable size
- Memory-mapped NumPy arrays for large datasets
- k-NN anomaly detection (finds most isolated images)
- Medical image quality validation (entropy, contrast, speckle noise)
- PHI-safe logging with SHA-256 hashing
- Performance metrics tracking

## Testing

```bash
python -m pytest tests/unit/ -v  # 68 tests covering all functionality
```

## Project Structure

```
Interview Project/
├── embedlab.py             # CLI interface (root level)
├── src/
│   ├── core/               # Business logic
│   │   ├── image_processor.py     # Medical image validation
│   │   ├── embedding_engine.py    # ResNet50 embeddings
│   │   ├── similarity_search.py   # Similarity & duplicates
│   │   └── index_manager.py       # Pipeline coordinator
│   └── utils/              # Support modules
│       ├── config.py       # Seed management
│       └── logging.py      # PHI-safe logging
├── tests/
│   └── unit/               # 68 comprehensive tests
├── assets/
│   ├── test_images/        # Sample ultrasounds for testing
│   └── queries/            # Query images
└── requirements.txt        # Dependencies
```

## Dependencies

- torch & torchvision - Deep learning backbone
- numpy - Numerical operations
- Pillow - Image processing
- scikit-learn - ML utilities
- click & rich - CLI interface
- pytest - Testing framework

## Performance

- Loading: 15-20ms per image
- Embedding: 150-200ms per image (CPU)
- Search: <5ms for 1000 images
- Dataset tested: 1,979 ultrasounds (989 valid after quality filtering)
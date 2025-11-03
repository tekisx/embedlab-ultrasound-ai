# Image Acquisition Plan - Medical Test Dataset

## Objective
Download ~20-30 real medical ultrasound images from public sources to create a diverse test dataset covering edge cases relevant to Ultrasound AI interview assessment.

## Target Image Collection

### Categories and Quantities

| Category | Count | Purpose | Edge Cases Covered |
|----------|-------|---------|-------------------|
| Normal Ultrasound | 5-10 | Baseline images | Standard preprocessing |
| Low Contrast/Dark | 2-3 | Exposure issues | Quality filter testing |
| Bright/Overexposed | 2-3 | Overexposure | Brightness detection |
| Near-Duplicates | 3-5 pairs | Consecutive frames | Duplicate detection |
| Different Modalities | 2-3 each | Fetal, cardiac, abdominal | Domain variation |
| Various Resolutions | 2-3 | Portrait/landscape/square | Aspect ratio handling |
| Color-Mapped | 2-3 | False-color ultrasound | RGB vs grayscale |
| With Text Overlays | 2-3 | Medical annotations | Text handling in images |

**Total Target: 20-30 images**

## Acquisition Strategy

### Phase 1: Search for Public Datasets

Use **WebSearch** to find publicly available ultrasound image sources:

**Search Queries:**
```
1. "public ultrasound image dataset"
2. "ultrasound images creative commons"
3. "kaggle ultrasound dataset"
4. "NIH medical ultrasound images"
5. "open source fetal ultrasound images"
6. "cardiac ultrasound dataset public"
7. "medical imaging dataset no PHI"
8. "ultrasound video frames dataset"
```

**Expected Sources:**
- Kaggle datasets (medical imaging competitions)
- NIH/NLM repositories
- Academic research data (Zenodo, FigShare, etc.)
- Hospital open data initiatives
- TCIA (The Cancer Imaging Archive)
- Medical imaging conferences with public data

### Phase 2: Scrape Dataset Pages

Use **Firecrawl** to extract direct image URLs from dataset pages:

**Process:**
1. Get dataset landing page URL from search
2. Use firecrawl_scrape or firecrawl_map to extract page structure
3. Identify direct image URLs or download links
4. Extract metadata (image descriptions, dimensions, modality)
5. Verify licensing (must be public domain, CC0, or research-use)

**Example Firecrawl Usage:**
```python
# Scrape dataset page for image URLs
firecrawl_scrape(
    url="https://example.com/ultrasound-dataset",
    formats=["markdown", "links"]
)
# Extract all image URLs from links
```

### Phase 3: Download Images

Use **Bash with curl** to download images:

**Process:**
1. Create download script with curl commands
2. Download images to temporary directory
3. Rename to meaningful filenames
4. Validate each image programmatically

**Example Download Commands:**
```bash
# Download with curl
curl -o temp_image_001.png "https://example.com/image1.png"

# Batch download with URLs from file
while read url; do
    curl -O "$url"
done < image_urls.txt
```

### Phase 4: Validate & Organize

**Validation Criteria:**
- File format: PNG or JPG
- File size: 10KB - 5MB (reasonable range)
- Dimensions: 50x50 to 4000x4000 pixels
- Loadable with PIL/Pillow (not corrupted)
- Contains actual image data (not blank)
- Visually appears to be medical/ultrasound

**Python Validation Script:**
```python
from PIL import Image
import numpy as np
from pathlib import Path

def validate_image(image_path: Path) -> bool:
    """Validate downloaded image meets criteria."""
    try:
        # Load image
        img = Image.open(image_path)

        # Check format
        if img.format not in ['PNG', 'JPEG']:
            return False

        # Check dimensions
        width, height = img.size
        if width < 50 or height < 50 or width > 4000 or height > 4000:
            return False

        # Check file size
        size_kb = image_path.stat().st_size / 1024
        if size_kb < 10 or size_kb > 5120:
            return False

        # Check not blank (mean intensity not near 0 or 255)
        arr = np.array(img)
        mean_val = arr.mean()
        if mean_val < 5 or mean_val > 250:
            return False

        return True
    except Exception:
        return False
```

**Organization:**
```
assets/images/
├── normal/
│   ├── ultrasound_fetal_001.png
│   ├── ultrasound_cardiac_002.png
│   └── ...
├── low_contrast/
│   ├── dark_exposure_001.png
│   └── ...
├── overexposed/
│   ├── bright_001.png
│   └── ...
├── duplicates/
│   ├── frame_001_a.png
│   ├── frame_001_b.png  # Near-duplicate of 001_a
│   └── ...
└── edge_cases/
    ├── text_overlay_001.png
    ├── color_mapped_001.png
    └── ...
```

## Specific Dataset Targets

### 1. Kaggle Ultrasound Datasets
**Search URL:** https://www.kaggle.com/datasets?search=ultrasound

**Likely Datasets:**
- Breast ultrasound images
- Fetal ultrasound datasets
- Thyroid ultrasound images

**Acquisition:**
- Use Kaggle API or direct download links
- Typically requires Kaggle account but images are public

### 2. NIH Open-i
**URL:** https://openi.nlm.nih.gov/

**Features:**
- Biomedical images from published research
- No PHI
- Direct image downloads

**Search Terms:**
- "ultrasound"
- "sonography"
- "echocardiography"

### 3. Cancer Imaging Archive (TCIA)
**URL:** https://www.cancerimagingarchive.net/

**Features:**
- Deidentified medical images
- Some ultrasound collections
- Requires simple registration

### 4. Academic Datasets (Zenodo, FigShare)
**Search:** "ultrasound dataset" site:zenodo.org

**Features:**
- Research datasets with DOIs
- Clear licensing
- Direct downloads

## Query Images Selection

From the downloaded images, select 6 for queries:

**Query Set Requirements:**
1. **Similar to Dataset**: 2 queries with clear matches in main dataset
2. **Near-Duplicates**: 2 queries that are near-duplicates of dataset images
3. **Edge Cases**: 1 query with unusual characteristics (very dark, very bright)
4. **Anomaly**: 1 query that's an outlier (different modality or artifact)

## Licensing Compliance

**Acceptable Licenses:**
- Public Domain (CC0)
- Creative Commons (CC BY, CC BY-SA)
- Academic research use
- Open Data Commons

**Must Avoid:**
- Images with "No Derivatives" restrictions
- Commercial-only licenses
- Images with visible patient information
- Images without clear licensing

## Automation Script Outline

```python
# image_acquisition.py
import requests
from pathlib import Path
from PIL import Image
import hashlib

class ImageAcquisition:
    """Automated medical image acquisition and validation."""

    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.downloaded = []

    def search_datasets(self, query: str) -> list[str]:
        """Search for datasets using WebSearch."""
        # Use Claude Code's WebSearch tool
        pass

    def scrape_image_urls(self, dataset_url: str) -> list[str]:
        """Extract image URLs from dataset page."""
        # Use Firecrawl tool
        pass

    def download_image(self, url: str, category: str) -> Path:
        """Download and validate single image."""
        # Use curl via Bash tool
        pass

    def validate_collection(self) -> dict:
        """Validate entire downloaded collection."""
        stats = {
            'total': len(self.downloaded),
            'valid': 0,
            'invalid': 0,
            'categories': {}
        }
        # Validation logic
        return stats

    def create_query_set(self, n: int = 6) -> list[Path]:
        """Select n images for query set."""
        # Selection logic
        pass
```

## Success Criteria

- ✓ Downloaded 20-30 medical ultrasound images
- ✓ All images validated (format, size, loadable)
- ✓ Diverse categories covering edge cases
- ✓ Clear licensing for all images
- ✓ Organized in logical directory structure
- ✓ 6 query images selected
- ✓ Near-duplicate pairs identified
- ✓ Documentation of image sources

## Fallback Strategy

If public datasets are insufficient:

1. **Generate Synthetic Medical-Style Images:**
   - Use noise patterns that simulate speckle
   - Apply transforms to create grayscale medical-looking images
   - Not ideal but ensures we have test data

2. **Use Smaller Curated Set:**
   - Focus on 10-15 highest-quality images
   - Ensure maximum diversity in smaller set
   - Supplement with minimal synthetic images

3. **Medical Image Generation Tools:**
   - Some open-source tools generate synthetic medical images
   - Example: StyleGAN-based medical image generators
   - Use only if real images unavailable

## Timeline Integration

This image acquisition should happen **during Phase 2.1** of the implementation plan (Testing & Quality Assurance phase), but can be done in parallel with early development phases to have images ready for integration testing.

## Notes

- Prioritize image diversity over quantity
- Document source URL for each image (for licensing audit)
- Maintain a metadata.json with image provenance
- Focus on ultrasound specifically (not CT, MRI, X-ray) for relevance to Ultrasound AI

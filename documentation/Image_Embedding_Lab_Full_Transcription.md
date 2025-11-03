# Image Embedding Lab - Full Transcription and Summary

## Page 1: Introduction and CLI Requirements
**File:** Page_1_Introduction_and_CLI_Requirements.jpg

### Full Transcription

**Image Embedding Lab — 2-Hour Build (CLI, no UI)**

Welcome! This is a focused, 2-hour exercise. You'll build a small CLI tool that computes image embeddings, runs nearest-neighbor search, and detects near-duplicates and anomalies — all without a UI. You may use any local libraries and AI assistants. Internet is allowed during your 2 hours. Our grader will run offline afterward.

**Goal**
Build a command-line tool that:
- Embeds images into fixed-length vectors.
- Performs nearest-neighbor search for any query images.
- Detects near-duplicates and anomalies using only the embeddings.

We will provide the assets and run your tool with our grader after time is up.

**Inputs**
- `./assets/images/` — ~120 PNG/JPG images (mixed domains; some near-duplicates; a few oddballs).
- `./assets/queries/` — 6 query images.
- (Optional convenience) `./assets/weights/` — pre-cached weights for a small model (e.g., torchvision ResNet18). Using other backbones is fine.

**Required CLI**
We'll call your tool using the following commands. Please match the interface exactly.

**Embed**
```
python embedlab.py embed --images-dir ./assets/images --out ./index
```
Compute embeddings for every image and persist whatever you need to `./index/` (npy, pickle, parquet—your choice). Must be deterministic for the same inputs. Log basic stats (image count, embedding dim, time).

**Search**
```
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json
```
For each query image, return top-5 most similar images with cosine similarity (or normalized dot-product) scores. Print one JSON object per query (to stdout).

Example output:
```json
{
  "query": "assets/queries/q1.png",
  "results": [
    {"path": "assets/images/017.png", "score": 0.88},
    {"path": "assets/images/044.png", "score": 0.86}
```

### Page 1 Summary
This page introduces a 2-hour technical interview challenge to build a CLI tool for image embedding and similarity search. The tool must compute image embeddings, perform nearest-neighbor searches, and detect duplicates/anomalies. The page specifies the directory structure for inputs (120 images, 6 queries) and exact CLI interface requirements for the `embed` and `search` commands, with specific output formats expected.

---

## Page 2: Functional Requirements and Deliverables
**File:** Page_2_Functional_Requirements_and_Deliverables.jpg

### Full Transcription

```
    ]
}
```

**Dedupe & Anomalies**
```
python embedlab.py analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json
```
Output JSON with:
- duplicate_groups: list of groups (each group ≥ 2 paths) representing near-duplicates using your threshold.
- anomalies: list of the top-N most isolated images (N = --anomaly-top), based on distance to neighbors (e.g., high mean distance to k-NN or low local density).

Example output:
```json
{
  "duplicate_groups": [
    ["assets/images/051.png","assets/images/063.png"],
    ["assets/images/022.png","assets/images/023.png","assets/images/028.png"]
  ],
  "anomalies": [
    "assets/images/004.png",
    "assets/images/099.png"
  ]
}
```

**Functional Requirements**
- Embeddings must come from a reasonable vision backbone (e.g., torchvision ResNet18/50, timm, or CLIP). CPU or GPU is fine.
- Use cosine similarity (or equivalent on normalized vectors) for search.
- Deterministic results (fixed seed).
- Reasonable runtime: embedding ≤ ~90s on the provided set on a typical laptop; search << 1s.

**Implicit Expectations (We Grade These)**
- Preprocessing: consistent resize/normalize; channels correct; no accidental grayscale bugs.
- Index structure: cache embeddings to disk; avoid recomputing on search/analyze.
- Numerics: normalize vectors if needed; guard against edge cases.
- Explainability (brief): log which backbone and dimensionality you used.

**Deliverables**
Submit a small repo or zip that includes:
- embedlab.py (or a small package) implementing embed, search, analyze.

### Page 2 Summary
This page completes the CLI specification with the `analyze` command for duplicate detection and anomaly identification. It outlines functional requirements including the use of vision models (ResNet/CLIP), performance expectations (≤90s embedding, <1s search), and implicit grading criteria covering preprocessing quality, index management, and numerical stability. The deliverables section begins, specifying submission format.

---

## Page 3: Optional Enhancements
**File:** Page_3_Optional_Enhancements.jpg

### Full Transcription

- Unit tests under tests/ (at least 3):
  - Embedding shape & determinism (seeded)
  - Search ranking stability on a tiny synthetic set
  - Duplicate grouping behavior at a threshold boundary
- README.md (how to run, backbone used, assumptions).
- Optional PROMPTS.md with 2–5 AI prompts that helped.

**Show-off Options (Optional Enhancements)**
You may add one or two enhancements below to demonstrate your strengths. We don't expect everything. Prioritize quality over quantity and document what you did in the README (what/why/impact).

**Quick Wins (≈15–30 min each)**
- Determinism & Repro: set seeds for random/NumPy/torch, log backbone name + embedding dim; add --seed and a short run manifest (manifest.json).
- Batching & Throughput: add --batch-size and measure images/sec; avoid recomputing embeddings on search/analyze.
- On-disk Index + Memory Map: persist embeddings in a single npy/parquet and memory-map for search.
- Explainability in JSON: include per-result reasons and distances; include backbone used.
- Type Hints + Lint/Format: mypy --strict, ruff/black.

**Solid Upgrades (≈30–60 min)**
- ANN for Speed (FAISS/hnswlib/Annoy) with optional --ann; compare latency vs exact.
- Duplicate Grouping via Union-Find over a k-NN graph; output clusters with ≥2 members.
- Anomaly Scoring that Scales: local density estimate; add --k and --anomaly-top flags.
- Embedding Compression: PCA to 128D or FP16 storage; report recall@5 impact.
- Similarity Calibration: learn a duplicate threshold from a small validation split; include ROC/PR snippets.

**Advanced Flex (≈60–120 min)**
- Multibackbone Ensemble (e.g., ResNet18 + Tiny-CLIP) with late-fusion of normalized similarities.
- Zero-shot Labels (if CLIP): optional text prompts ("blue button", "login screen") via --q-text mode.
- Tiered Retrieval: ANN coarse pass → exact rerank.
- Adaptive Thresholding for Duplicates: per-cluster thresholds using neighbor stats (median + MAD).
- Quality & Safety Filters: flag corrupted/blank/low-entropy images and exclude from index.
- Observability: emit metrics.json with timing and counts.
- Packaging & CLI polish: package entry points and helpful --help.

### Page 3 Summary
This page details optional enhancements categorized by difficulty and time investment. "Quick Wins" (15-30 min) include determinism improvements, batching, and type hints. "Solid Upgrades" (30-60 min) feature ANN implementations, advanced duplicate detection, and compression techniques. "Advanced Flex" (60-120 min) options include multi-model ensembles, zero-shot capabilities with CLIP, tiered retrieval systems, and sophisticated filtering/monitoring features.

---

## Page 4: Final Tips and Reminders
**File:** Page_4_Final_Tips_and_Reminders.jpg

### Full Transcription

- Re-ranking by Visual Diversity: encourage diverse top-k while keeping relevance.

**One-liner to Keep in the Spec**
You're invited to add one or two optional enhancements (see "Show-off options"). We don't expect everything—demonstrate your best judgment, explain your tradeoffs, and, if possible, quantify the impact (accuracy/latency/robustness).

**Friendly Reminders**
- No UI required; keep everything CLI-driven.
- If you pull a model during the 2 hours, cache artifacts to disk; grader runs offline.
- Pick one or two extras. Depth > breadth.

### Page 4 Summary
The final page concludes with the last advanced option (visual diversity re-ranking) and provides important reminders. It emphasizes that optional enhancements should be chosen strategically (1-2 max, depth over breadth), all work must be CLI-driven with no UI, and any models downloaded during the exercise should be cached for offline grading. The page reinforces the importance of explaining tradeoffs and quantifying impact of chosen enhancements.

---

## Overall Assignment Summary

This is a comprehensive 2-hour technical interview assignment for building an image embedding and similarity search CLI tool. The core requirements involve:

1. **Core Functionality**: Build a Python CLI tool that can embed images, search for similar images, and detect duplicates/anomalies
2. **Three Required Commands**: `embed`, `search`, and `analyze` with specific input/output formats
3. **Performance Targets**: Embedding ~120 images in ≤90s, search in <1s
4. **Deliverables**: Python implementation, unit tests, README, optional AI prompts documentation
5. **Optional Enhancements**: Ranging from quick improvements (15-30 min) to advanced features (60-120 min) to showcase expertise

The assignment tests practical ML engineering skills including model selection, vector similarity search, efficient indexing, duplicate detection algorithms, anomaly detection, and code quality practices. It emphasizes deterministic, performant solutions with proper documentation and testing.
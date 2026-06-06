# Graph Report - /sessions/charming-sweet-edison/mnt/neurolens  (2026-06-05)

## Corpus Check
- 10 files · ~5,000 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 70 nodes · 155 edges · 5 communities (4 shown, 1 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core ML Library|Core ML Library]]
- [[_COMMUNITY_Gradio Web Application|Gradio Web Application]]
- [[_COMMUNITY_Training & Experimentation|Training & Experimentation]]
- [[_COMMUNITY_Infrastructure & DevOps|Infrastructure & DevOps]]
- [[_COMMUNITY_External Dependencies|External Dependencies]]

## God Nodes (most connected - your core abstractions)

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (5 total, 1 thin omitted)

### Community 0 - "Core ML Library"
Cohesion: 1.00
Nodes (29): src/__init__.py, src/model.py, src/dataset.py, src/inference.py, BACKBONE_NAME, NUM_CLASSES, create_classifier(), restore_from_checkpoint() (+21 more)

### Community 1 - "Gradio Web Application"
Cohesion: 1.00
Nodes (13): load_model(), coerce_to_pil(), confidence_band(), empty_response(), classify(), discover_examples(), MODEL, CAM_ENGINE (+5 more)

### Community 2 - "Training & Experimentation"
Cohesion: 1.00
Nodes (9): Data Loading & EDA, Patient-Level Splitting, Stage 1 Fine-Tuning, Stage 2 Fine-Tuning, Model Evaluation, Grad-CAM Visualisation, Failure Analysis, Patient-Level Split (+1 more)

### Community 4 - "External Dependencies"
Cohesion: 1.00
Nodes (12): torch, torchvision, timm, pytorch_grad_cam, gradio, Pillow, numpy, pandas (+4 more)

## Knowledge Gaps
- **7 isolated node(s):** `src/__init__.py`, `assets/logo.svg`, `samples/README.md`, `PathLike`, `EVAL_TRANSFORM` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.
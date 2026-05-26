<p align="center">
  <img src="assets/logo.svg" alt="NeuroLens logo" width="140"/>
</p>

<h1 align="center">NeuroLens</h1>

<p align="center">
  <em>Honest, leakage-free brain MRI tumor classification with explainable predictions.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/pytorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/backbone-EfficientNet--B3-009688.svg" alt="EfficientNet-B3">
  <img src="https://img.shields.io/badge/test%20accuracy-95.05%25-brightgreen.svg" alt="Test accuracy 95.05%">
  <img src="https://img.shields.io/badge/macro%20AUC-0.9965-success.svg" alt="Macro AUC 0.9965">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

---

## Overview

**NeuroLens** is a 4-class brain MRI tumor classifier built on raw Figshare and Br35H data with **strict patient-level splitting**. It distinguishes between **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**, and produces a **Grad-CAM attention map** for every prediction so the reasoning behind a classification is visible — not hidden behind a probability.

The project deliberately targets an **honest** accuracy number rather than the inflated 96–99% numbers commonly reported on this dataset. By splitting at the patient level instead of the image level, NeuroLens eliminates the data leakage that quietly boosts most public implementations by 5–15 percentage points.

| Metric | Value |
|---|---|
| Test accuracy (TTA) | **95.05%** |
| Test accuracy (single-pass) | 94.91% |
| Macro F1 | 0.9364 |
| Macro AUC-ROC | **0.9965** |
| Test set size | 687 unseen patient images |
| Patient leakage between splits | **0** (verified by set intersection) |

---

## Why this project is different

Most public brain tumor classifiers report 96–99% test accuracy on the same dataset. **Those numbers are inflated by data leakage.**

The Figshare source contains 3,064 MRI slices from only **233 unique patients** — roughly 13 slices per patient. A naive random split at the image level scatters slices from the same patient into both the training and test sets. The model memorises patient-specific anatomy (skull shape, tissue contrast, scanner artifacts) rather than learning generalisable tumor features. The headline accuracy looks great; the underlying learning is poor.

NeuroLens implements **patient-level splitting**, validated by explicit set-intersection checks:

```text
Train ∩ Val  = 0 patients
Train ∩ Test = 0 patients
Val   ∩ Test = 0 patients
```

The 95.05% accuracy reported here is computed on patients the model has genuinely never encountered — a number that is **lower** than what naive implementations claim, but is the one that actually reflects real-world generalisation.

---

## Live demo

▶ **Try NeuroLens on Hugging Face Spaces:** [Launch demo →](https://huggingface.co/spaces/Megalodon55681/NeuroLens)

> *Hugging Face Spaces may take a few seconds to warm up on first request — the container sleeps when idle.*

---

## Dataset

NeuroLens combines two raw public sources rather than relying on a pre-aggregated dataset, because the aggregated versions strip the patient identifiers needed for honest splitting.

| Source | Classes contributed | Images | Patients |
|---|---|---|---|
| Figshare (Cheng et al., 2017) | Glioma · Meningioma · Pituitary | 3,064 | 233 |
| Br35H Brain Tumor Detection | No tumor | 1,500 | 1,500 (synthetic IDs) |
| **Combined corpus** | **4 classes** | **4,564** | — |

### Class distribution

| Class | Count | Share |
|---|---|---|
| No tumor | 1,500 | 32.9% |
| Glioma | 1,426 | 31.2% |
| Pituitary | 930 | 20.4% |
| Meningioma | 708 | 15.5% |

<p align="center">
  <img src="outputs/class_distribution.png" alt="Class distribution" width="600"/>
</p>

### Why combine the sources manually?

Figshare provides three tumor types with patient IDs, which is essential for patient-level splitting. Br35H provides healthy controls (no-tumor) that Figshare lacks. Aggregated datasets that combine these sources strip the patient IDs in the process, making proper splitting impossible. Combining the raw sources by hand was the only way to keep all four classes under a single, honest splitting methodology.

**Documented limitation:** Br35H does not provide patient IDs. Each Br35H image is treated as an independent pseudo-patient, so the no-tumor split is effectively image-level while the tumor splits are patient-level. This asymmetry is disclosed openly here rather than buried.

### Slices per patient — motivating the split strategy

<p align="center">
  <img src="outputs/slices_per_patient.png" alt="Slices per patient" width="600"/>
</p>

Many Figshare patients contribute 15–25 slices. Without patient-level grouping, those slices would scatter across splits and the model would silently learn patient identities instead of tumor features.

---

## Methodology

### Splitting strategy

* `StratifiedGroupKFold` from scikit-learn — preserves class balance **and** patient grouping.
* Two-stage split: carve out 15% test (`n_splits=7`), then split the remainder into train + 15% val.
* Final ratios: **71.6% train / 13.3% val / 15.1% test**.
* Zero patient leakage verified via explicit set intersections.
* The original Cheng et al. cross-validation folds (`cvind.mat`) were inspected but not used; an independent patient-level split was implemented so all four classes — including Br35H's no-tumor — fall under a single methodology.

<p align="center">
  <img src="outputs/split_proportions.png" alt="Split proportions" width="600"/>
</p>

### Preprocessing

Each image goes through a one-time preprocessing step saved as a PNG:

* Per-image min-max normalisation (handles MRI intensity variation across scanners).
* Grayscale replicated to 3 channels (compatibility with ImageNet-pretrained backbones).
* Stored as `uint8` PNG, indexed by row number and class label.

Resizing to 224 × 224 happens inside the DataLoader rather than during preprocessing, so the input resolution can be experimented with without regenerating the entire image cache.

### Model

| Property | Value |
|---|---|
| Backbone | EfficientNet-B3 (`timm`) |
| Pretrained on | ImageNet |
| Classification head | 4-class linear layer (replaces original 1000-class head) |
| Total parameters | ~10.7 M |

### Two-stage transfer learning

**Stage 1 — Frozen backbone (5 epochs):**
* Train only the classifier head (~6,148 parameters).
* AdamW, learning rate 1e-3.
* Result: validation accuracy 81.74%.

**Stage 2 — Full fine-tuning (17 epochs, early-stopped):**
* Unfreeze the entire network.
* Differential learning rates: backbone 5e-5, head 1e-4.
* AdamW with weight decay 1e-4, cosine annealing schedule, gradient clipping at `max_norm=1.0`.
* Early stopping with patience 8.
* Result: validation accuracy 94.57%.

<p align="center">
  <img src="outputs/combined_training_curves.png" alt="Training curves" width="700"/>
</p>

### Data augmentation (training only)

* Random rotation up to ±15°.
* Small random translation (±5%) and scale (95–105%).
* Colour jitter (brightness ±20%, contrast ±20%).
* **No horizontal flip** — brain MRI has anatomical asymmetry that flipping would corrupt.

<p align="center">
  <img src="outputs/augmentation_examples.png" alt="Augmentation examples" width="700"/>
</p>

### Loss function

Weighted cross-entropy with inverse-frequency class weights, compensating for class imbalance. Meningioma — the smallest class — receives the largest weight.

### Test-time augmentation (TTA)

At inference, each test image is passed through the model five times with mild augmentations (rotation ±5°, small brightness/contrast shifts) and the softmax probabilities are averaged before `argmax`. TTA contributed a measurable but modest **+0.14%** accuracy improvement on top of the single-pass baseline.

---

## Results

### Headline numbers

| Metric | Value |
|---|---|
| Test accuracy (TTA) | **95.05%** |
| Test accuracy (single-pass) | 94.91% |
| Macro AUC-ROC | **0.9965** |
| Macro F1 | 0.9364 |
| Test images | 687 (patient-level held-out) |

### Per-class performance (TTA)

<p align="center">
  <img src="outputs/per_class_performance.png" alt="Per-class performance" width="700"/>
</p>

<p align="center">
  <img src="outputs/confusion_matrix_tta.png" alt="Confusion matrix (TTA)" width="500"/>
</p>

<p align="center">
  <img src="outputs/roc_curves_tta.png" alt="ROC curves (TTA)" width="600"/>
</p>

### The honest weakness — meningioma

Meningioma is the model's hardest class. Of 34 total test errors, **20 are meningioma misclassifications**, broken down as:

* **Predicted as glioma (12 cases):** The most common confusion. Both tumor types can present as a hyperintense mass on T1-weighted MRI; without multi-modal input the visual cues are genuinely overlapping.
* **Predicted as pituitary (8 cases):** These meningiomas were predominantly located in the lower-middle brain region — the same anatomical area where pituitary tumors occur. Tumor size varied, ruling size out as the primary confounder; **location** appears to be the driving factor.

A clinically reassuring observation: **the model never confused a meningioma for no-tumor.** The error mode is tumor *subtype* confusion, not tumor *presence*.

<p align="center">
  <img src="outputs/meningioma_failures.png" alt="Meningioma failure cases" width="700"/>
</p>

### High-confidence failures (the dangerous mode)

Ten of the 34 errors were made with >90% confidence — split exactly 5–5 between two confusion pairs:

* 4 of 5 high-confidence meningioma errors → predicted as pituitary.
* 4 of 5 high-confidence glioma errors → predicted as meningioma.
* 0 high-confidence errors involved the pituitary class.

Visual inspection suggests these images are **genuinely ambiguous**, not obvious failures. Several show unusual contrast or darker-than-typical brightness, suggesting image quality variation contributes alongside the underlying class similarity. In a clinical setting these would warrant secondary review regardless of model confidence.

<p align="center">
  <img src="outputs/high_confidence_failures.png" alt="High-confidence failures" width="700"/>
</p>

### Calibration

The model exhibits reasonable confidence calibration:

* Mean confidence on **correct** predictions: 98.3%
* Mean confidence on **incorrect** predictions: 76.3%
* Clean ~22 percentage-point separation between right and wrong.

This means confidence could serve as a reliable triage signal in deployment — low-confidence predictions can be routed to human review.

<p align="center">
  <img src="outputs/calibration_analysis.png" alt="Calibration analysis" width="700"/>
</p>

---

## Interpretability — Grad-CAM

To verify that the model classifies based on genuine tumor features rather than dataset shortcuts (scanner artifacts, watermarks, positioning), **Grad-CAM** (Gradient-weighted Class Activation Mapping) was applied across the entire test set. Grad-CAM produces a heatmap highlighting the image regions that most influenced a prediction.

<p align="center">
  <img src="outputs/gradcam_overview.png" alt="Grad-CAM overview" width="800"/>
</p>

### What the heatmaps reveal

Localisation quality varies systematically by class:

* **Meningioma — strongest localisation.** Heatmaps land directly on the tumor region. This is notable because meningioma is also the model's hardest class — even when it misclassifies, it is generally looking at the right place.
* **Pituitary — tight, focal heatmaps.** Attention concentrates on the small sellar region where pituitary tumors occur. The small footprint reflects the small anatomical structure, not a defect.
* **Glioma and no-tumor — diffuse central attention.** Heatmaps spread across the central brain rather than tightly localising. For no-tumor this is expected; for glioma it suggests the model uses broader contextual features alongside the lesion itself.

### Two distinct failure modes

Grad-CAM on misclassified cases revealed that the model fails in two fundamentally different ways:

1. **Correct attention, wrong class.** Some meningioma → glioma errors have heatmaps that correctly cover the tumor, but the model still picks the wrong class. The tumor's heterogeneous internal texture genuinely misleads the classifier. This is a feature-learning limitation, not an attention failure.

2. **Wrong attention, wrong class.** Some glioma → meningioma errors have heatmaps that do not cover the tumor at all — the model made a confident decision based on regions outside the lesion. This points to distractor features or image-quality effects.

This distinction matters because the two problems need different fixes:
the first calls for **richer features** (e.g., multi-modal MRI);
the second calls for **attention regularisation** or **input-quality screening**.

---

## Limitations

1. **Single MRI modality.** Only T1-weighted contrast-enhanced MRI was used. Clinical practice uses multiple modalities (T1, T2, FLAIR, T1ce). The meningioma–glioma confusion is precisely the kind of error that multi-modal input typically resolves, because different sequences highlight different tissue properties.

2. **Meningioma class weakness.** 19.6% error rate on meningioma is the dominant failure mode. Class-weighted loss partially addressed the sample imbalance (708 meningioma vs 1,426 glioma training images) but did not eliminate it. The underlying constraint is the number of unique meningioma patients.

3. **2D slice-based classification.** Each prediction is on a single slice in isolation. Clinical radiologists interpret 3D volumes. Slice-based models can miss spatial context.

4. **Br35H lacks patient IDs.** No-tumor images are split image-level while tumor images are split patient-level — a known asymmetry, documented but not resolved.

5. **Dataset distribution.** Figshare data was collected from two specific hospitals in China. Performance on MRI from different scanners, acquisition protocols, or patient populations may degrade. No cross-scanner validation was performed.

6. **Not for clinical use.** This is a portfolio and research demonstration. It has not been validated in a clinical setting, has not been reviewed by radiologists, and **must not** be used for any medical decision-making.

---

## Roadmap (v2)

Planned improvements being explored for a future release:

* **Multi-modal MRI** using BraTS (T1, T2, FLAIR, T1ce together). Most likely path to fixing the meningioma weakness.
* **Focal loss or class-balanced sampler** as an alternative to weighted cross-entropy — targets hard-to-classify samples explicitly.
* **MixUp / CutMix augmentation** for minority-class robustness.
* **3D model (3D ResNet or volumetric U-Net)** to use spatial context from neighbouring slices.
* **k-fold cross-validation** for tighter accuracy confidence intervals.
* **Monte Carlo Dropout** for principled uncertainty quantification at inference time.
* **ONNX export + ONNX Runtime benchmarks** for faster, framework-independent deployment.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch 2.x |
| Model zoo | `timm` (PyTorch Image Models) |
| Explainability | `pytorch-grad-cam` |
| Demo UI | Gradio |
| Hosting | Hugging Face Spaces |
| Notebook environment | Google Colab (Tesla T4) |

---

## Project structure

```text
neurolens/
│
├── README.md                       Project overview (this file)
├── LICENSE                         MIT license
├── requirements.txt                Python dependencies
├── .gitignore                      Excludes data and checkpoints
├── app.py                          Gradio demo application
│
├── assets/
│   └── logo.svg                    Project logo
│
├── src/
│   ├── __init__.py
│   ├── model.py                    EfficientNet-B3 construction utilities
│   ├── dataset.py                  Dataset class and image transforms
│   └── inference.py                Prediction and Grad-CAM utilities
│
├── notebooks/
│   └── brain_tumor_full_pipeline.ipynb   End-to-end training notebook
│
├── samples/
│   └── README.md                   Curated example images for the demo
│
└── outputs/                        Generated artefacts
    ├── training curves, confusion matrices, ROC curves
    ├── Grad-CAM visualisations
    ├── failure analysis plots
    ├── per_class_metrics.csv, evaluation_summary.txt
    └── screenshots/
```

---

## Reproduce locally

### 1. Clone

```bash
git clone https://github.com/TheMEGALODON55681/NeuroLens.git
cd NeuroLens
```

### 2. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

The project was developed on Google Colab with PyTorch 2.x, Python 3.12, and a Tesla T4 GPU. A GPU is recommended for training but **not required for inference**.

### 3. Download the data

The datasets are not bundled with this repository. Download them from the original sources:

* **Figshare Brain Tumor Dataset** — Cheng et al., 2017:
  [https://figshare.com/articles/dataset/brain_tumor_dataset/1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
* **Br35H Brain Tumor Detection** (no-tumor class) — Ahmed Hamada, 2020:
  [https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

After downloading, place the raw files following the directory structure expected at the top of the training notebook.

### 4. Train

Open `notebooks/brain_tumor_full_pipeline.ipynb` and run the sections in order. The notebook covers acquisition, preprocessing, patient-level splitting, training, evaluation, and Grad-CAM generation. Processed data is regenerated at runtime rather than downloaded.

### 5. Run the demo without training

To skip training, download the pretrained checkpoint:

* Download `stage2_best.pt` from the Hugging Face Space (Files tab):
  [https://huggingface.co/spaces/Megalodon55681/NeuroLens/tree/main](https://huggingface.co/spaces/Megalodon55681/NeuroLens/tree/main)
* Place it in the project root next to `app.py`.
* Launch the local demo:

  ```bash
  python app.py
  ```

The Gradio interface will be available at `http://127.0.0.1:7860`.

---

## Acknowledgements

* **Cheng et al. (2017)** — original Figshare brain tumor dataset (Cheng, Jun (2017). brain tumor dataset. figshare. Dataset).
* **Br35H** — no-tumor MRI dataset by Ahmed Hamada (2020).
* **Ross Wightman / `timm`** — PyTorch Image Models library providing the EfficientNet-B3 implementation and pretrained ImageNet weights.
* **Jacob Gildenblat / `pytorch-grad-cam`** — implementation used for the attention overlays.

---

## Contact

**Aryan Sharma**

* GitHub: [@TheMEGALODON55681](https://github.com/TheMEGALODON55681)
* Email: [aryansharma10011@gmail.com](mailto:aryansharma10011@gmail.com)

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for full text.

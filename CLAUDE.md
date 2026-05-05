# Agent Handoff — EuroSAT Land-Cover Classifier

This document is the authoritative context file for any agent continuing this project.
Read it fully before touching any code.

---

## What This Project Is

A computer vision pipeline that classifies **land use / land cover** from **Sentinel-2 satellite imagery**
using transfer learning on the public **EuroSAT RGB** dataset.

- **Dataset**: 27,000 labelled 64×64 Sentinel-2 RGB patches across 10 land-cover classes
- **Model**: EfficientNet-B0 pretrained on ImageNet-1K, fine-tuned end-to-end
- **GitHub**: https://github.com/abTuring13/satellite-landuse-classifier

---

## Environment

| Item | Value |
|---|---|
| Machine | MacBook Air — Apple Silicon (MPS available) |
| Python | 3.11 (`/usr/local/bin/python3.11`) |
| Run scripts with | `python3.11 src/<script>.py` |
| Key packages | torch 2.5.1, torchvision, numpy, matplotlib, seaborn, scikit-learn, datasets, huggingface-hub, Pillow, tqdm |
| Install missing deps | `pip3.11 install -r requirements.txt` |
| GitHub CLI | `/opt/homebrew/bin/gh` — authenticated as `abTuring13` |

---

## Repository Layout

```
satellite-landuse-classifier/
├── src/
│   ├── config.py          # All hyperparameters and path constants
│   ├── dataset.py         # HuggingFace download, splits, augmentation, DataLoaders
│   ├── model.py           # EfficientNet-B0 head replacement + device helper
│   ├── train.py           # Training loop: AdamW + OneCycleLR + label smoothing
│   ├── evaluate.py        # Confusion matrix, classification report, history plots
│   ├── visualize.py       # Class sample grid, prediction overlays, class distribution
│   ├── main.py            # End-to-end entry point (train + evaluate + visualize)
│   ├── predict.py         # CLI inference on arbitrary image files
│   └── export_samples.py  # Exports PNG patches from Arrow cache for testing
├── results/
│   ├── best_model.pt              # Trained checkpoint (16 MB) — DO NOT DELETE
│   ├── history.json               # Per-epoch loss/acc for all 15 epochs
│   ├── metrics.json               # Final test accuracy
│   ├── classification_report.txt  # Per-class precision/recall/F1
│   ├── class_samples.png
│   ├── class_distribution_train.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── predictions.png
│   └── sample_images/             # 15 exported PNGs (one per class) for predict.py demos
├── data/                  # HuggingFace Arrow cache (~166 MB) — gitignored
├── requirements.txt
├── README.md
└── CLAUDE.md              # ← this file
```

---

## Trained Model — Performance Summary

Training completed: **15 epochs**, Apple MPS, ~19 hours ago.

| Split | Accuracy |
|---|---|
| Train | 99.15% |
| Validation | 98.36% (best: 98.56%) |
| **Test** | **98.68%** |

### Per-class Test Results (2,430 samples)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | 0.9722 | 0.9929 | 0.9825 |
| Forest | 0.9964 | 1.0000 | 0.9982 |
| HerbaceousVegetation | 0.9853 | 0.9746 | 0.9800 |
| Highway | 0.9761 | 0.9808 | 0.9784 |
| Industrial | 0.9915 | 0.9957 | 0.9936 |
| Pasture | 0.9888 | 0.9620 | 0.9752 |
| PermanentCrop | 0.9783 | 0.9740 | 0.9761 |
| Residential | 1.0000 | 0.9922 | 0.9961 |
| River | 0.9803 | 0.9851 | 0.9827 |
| SeaLake | 0.9965 | 1.0000 | 0.9982 |
| **macro avg** | **0.9865** | **0.9857** | **0.9861** |

Weakest class: **Pasture** (F1 0.9752) — commonly confused with HerbaceousVegetation and AnnualCrop.

---

## How to Run

```bash
cd "satellite-landuse-classifier"

# Full pipeline: downloads data, trains, evaluates, saves plots
python3.11 src/main.py

# Evaluate only using saved checkpoint (no re-training)
python3.11 src/main.py --eval

# Export sample PNGs from cached dataset for testing
python3.11 src/export_samples.py --n 10

# Predict on any image file(s)
python3.11 src/predict.py results/sample_images/Forest_00875.png
python3.11 src/predict.py img1.png img2.png --top 5

# Override epochs (e.g. for quick smoke-test)
python3.11 src/main.py --epochs 3
```

---

## Dataset Notes

- Source: HuggingFace Hub — `blanchon/EuroSAT_RGB`
- Cached locally at `data/` (~166 MB Arrow files) — already downloaded, no re-download needed
- Loaded via `datasets` library; converted to PyTorch Dataset in `src/dataset.py`
- Splits: 70% train / 15% val / 15% test, seeded with `RANDOM_SEED=42`
- The `data/` folder is gitignored — a new machine must re-download (~90 MB zip → 166 MB unpacked)

---

## Key Design Decisions (context for future changes)

| Decision | Rationale |
|---|---|
| EfficientNet-B0 over ResNet | Faster convergence, better accuracy/param ratio at small image sizes |
| OneCycleLR scheduler | Outperforms StepLR/CosLR on small datasets; reaches peak perf in fewer epochs |
| Label smoothing = 0.1 | Prevents overconfidence; marginal improvement on val accuracy |
| Full fine-tuning (no frozen backbone) | EuroSAT is spectrally different from ImageNet; backbone adaptation is necessary |
| MPS / CUDA / CPU auto-detection | `get_device()` in `model.py` handles all three transparently |
| Arrow cache in `data/` | Avoids repeated downloads; gitignored because it's large |

---

## Suggested Next Steps

The model is production-quality at 98.68% test accuracy. Natural extensions, roughly ordered by effort:

### Low effort
- **Grad-CAM visualizations** — show which pixels the model focuses on per prediction.
  Add `pytorch-grad-cam` to requirements and a `src/gradcam.py` module.
- **Upload `best_model.pt` to HuggingFace Hub** — make the checkpoint publicly downloadable
  so `predict.py` works on a fresh machine without training. Use `huggingface_hub.upload_file`.
- **Streamlit demo app** — `src/app.py` with image upload → prediction display.
  `pip install streamlit`, run with `streamlit run src/app.py`.

### Medium effort
- **Multispectral (13-band) support** — swap to the EuroSAT MS variant (`blanchon/EuroSAT_MS`).
  Requires changing the input conv layer from 3 → 13 channels and retraining.
- **Geospatial inference on real GeoTIFF files** — use `rasterio` to tile a full satellite scene
  and run a sliding-window prediction, producing a land-cover map with colour overlay.
- **Model export** — export `best_model.pt` to ONNX (`torch.onnx.export`) for deployment.

### Larger effort
- **Object detection** — move from classification to detection using the DOTA or xView datasets.
  Replace EfficientNet backbone with a YOLO or DETR head.
- **Change detection** — compare two time-series images of the same tile to detect deforestation,
  urban sprawl, etc. Requires a Siamese or difference-based architecture.
- **FastAPI inference service** — wrap `predict.py` logic in a REST endpoint.
  `POST /predict` accepts an image, returns JSON with class probabilities.

---

## Git History (as of handoff)

```
ea11d8a  Add export_samples.py utility to extract PNG test images from Arrow cache
2e79608  Add training results, visualizations, and sample images
b1b814e  Initial commit: EuroSAT land-cover classifier with EfficientNet-B0
```

All three commits are on `main`, fully pushed to `origin`.

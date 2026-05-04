# Satellite Land Use / Land Cover Classification

Computer vision pipeline for classifying land cover types from Sentinel-2 satellite imagery using transfer learning with **EfficientNet-B0**.

## Dataset — EuroSAT

[EuroSAT](https://github.com/phelber/EuroSAT) is a public benchmark dataset built from **Sentinel-2 satellite images** covering 13 European countries. This project uses the RGB variant.

| Property | Value |
|---|---|
| Images | 27,000 |
| Resolution | 64 × 64 px (10m/px) |
| Classes | 10 land-cover categories |
| Source | Copernicus Sentinel-2 program |

### Classes

| # | Class | Description |
|---|---|---|
| 0 | AnnualCrop | Annual crop fields |
| 1 | Forest | Dense forest cover |
| 2 | HerbaceousVegetation | Grasslands and shrublands |
| 3 | Highway | Roads and motorways |
| 4 | Industrial | Industrial and commercial zones |
| 5 | Pasture | Permanent pasture land |
| 6 | PermanentCrop | Vineyards, orchards, olive groves |
| 7 | Residential | Urban residential areas |
| 8 | River | Rivers and canals |
| 9 | SeaLake | Seas, lakes, water bodies |

## Model

- **Architecture**: EfficientNet-B0 (pretrained on ImageNet-1K)
- **Strategy**: Full fine-tuning with label smoothing + OneCycleLR scheduler
- **Augmentations**: Random H/V flips, rotation ±15°, colour jitter
- **Hardware**: Automatically detects CUDA / Apple MPS / CPU

## Results

| Split | Accuracy |
|---|---|
| Train | ~97% |
| Validation | ~96% |
| **Test** | **~96%** |

> Results after 15 epochs on the full 27k dataset.

### Visualizations

After training, the following plots are saved to `results/`:

| File | Description |
|---|---|
| `class_samples.png` | Sample satellite patches per class |
| `class_distribution_train.png` | Training set class balance |
| `training_history.png` | Loss and accuracy curves |
| `confusion_matrix.png` | Per-class confusion (counts + normalized) |
| `predictions.png` | 20 test-set images with true vs predicted labels |

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train + evaluate (downloads dataset automatically ~90MB)
python src/main.py

# Evaluate only with saved checkpoint
python src/main.py --eval

# Predict on your own image
python src/predict.py path/to/satellite_image.jpg
python src/predict.py img.png --top 5
```

### Options

```
python src/main.py --epochs 10   # override training epochs
python src/main.py --freeze      # freeze EfficientNet backbone (faster, slightly lower accuracy)
```

## Project Structure

```
satellite-landuse-classifier/
├── src/
│   ├── config.py       # Hyperparameters and paths
│   ├── dataset.py      # EuroSAT loading + augmentation
│   ├── model.py        # EfficientNet-B0 head replacement
│   ├── train.py        # Training loop with OneCycleLR
│   ├── evaluate.py     # Metrics, confusion matrix, history plots
│   ├── visualize.py    # Dataset samples + prediction overlays
│   ├── main.py         # Entry point
│   └── predict.py      # Inference on arbitrary images
├── results/            # Saved plots, metrics, model checkpoint
├── requirements.txt
└── README.md
```

## References

- Helber, P. et al. (2019). **EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification**. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*. [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)
- Tan, M. & Le, Q.V. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Dataset hosted on HuggingFace Hub: [`blanchon/EuroSAT_RGB`](https://huggingface.co/datasets/blanchon/EuroSAT_RGB)

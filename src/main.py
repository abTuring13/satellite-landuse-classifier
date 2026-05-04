#!/usr/bin/env python3
"""
EuroSAT Land Use / Land Cover Classification
Sentinel-2 satellite imagery — EfficientNet-B0 transfer learning

Usage:
    python src/main.py          # full train + evaluate
    python src/main.py --eval   # evaluate only (requires best_model.pt)
"""
import sys
import argparse
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import RESULTS_DIR, DATA_DIR
from dataset import load_eurosat, build_dataloaders
from model import build_model, get_device
from train import train
from evaluate import run_evaluation
from visualize import plot_class_samples, plot_predictions, plot_class_distribution


def parse_args():
    p = argparse.ArgumentParser(description="EuroSAT land-cover classifier")
    p.add_argument("--eval", action="store_true", help="Skip training, evaluate saved model")
    p.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    p.add_argument("--freeze", action="store_true", help="Freeze EfficientNet backbone")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.epochs:
        import config
        config.NUM_EPOCHS = args.epochs

    ds = load_eurosat(cache_dir=DATA_DIR)
    train_loader, val_loader, test_loader = build_dataloaders(ds)

    model = build_model(freeze_backbone=args.freeze)
    device = get_device()

    print(f"\n{'='*60}")
    print("EuroSAT Land Use / Land Cover Classifier")
    print(f"  Dataset : EuroSAT RGB (Sentinel-2, 10 classes)")
    print(f"  Model   : EfficientNet-B0 (ImageNet pretrained)")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    # Visualize dataset before training
    print("Plotting dataset samples and class distribution...")
    plot_class_samples(train_loader)
    plot_class_distribution(train_loader, split_name="train")

    best_model_path = RESULTS_DIR / "best_model.pt"
    history = None

    if not args.eval:
        model, history = train(model, train_loader, val_loader)
    else:
        if not best_model_path.exists():
            print(f"ERROR: No saved model at {best_model_path}. Run without --eval first.")
            sys.exit(1)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded model from {best_model_path}")

        history_path = RESULTS_DIR / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

    # Always load best checkpoint for evaluation
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\nRunning evaluation on test set...")
    run_evaluation(model, test_loader, history=history)

    print("\nGenerating prediction visualizations...")
    model = model.to(device)
    plot_predictions(model, test_loader, device)

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

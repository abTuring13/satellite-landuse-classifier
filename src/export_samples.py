#!/usr/bin/env python3
"""
Export a few test images from the EuroSAT Arrow cache as PNG files
so they can be passed to predict.py.

Usage:
    python3 src/export_samples.py          # saves 10 random samples
    python3 src/export_samples.py --n 20   # save 20 samples
"""
import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, CLASSES, RESULTS_DIR
from dataset import load_eurosat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples to export")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "sample_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_eurosat(cache_dir=DATA_DIR)
    split = ds.get("test") or ds.get("validation") or ds["train"]

    indices = random.sample(range(len(split)), min(args.n, len(split)))
    saved = []
    for idx in indices:
        item = split[idx]
        img = item["image"]
        label = CLASSES[item["label"]]
        fname = out_dir / f"{label}_{idx:05d}.png"
        img.save(fname)
        saved.append(fname)
        print(f"  Saved: {fname.name}  (true class: {label})")

    print(f"\n{len(saved)} images saved to: {out_dir}")
    print("\nRun predictions with:")
    print(f"  python3 src/predict.py {' '.join(str(p) for p in saved[:3])} ...")

if __name__ == "__main__":
    main()

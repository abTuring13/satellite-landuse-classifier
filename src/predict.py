#!/usr/bin/env python3
"""
Predict the land-cover class of one or more satellite image files.

Usage:
    python src/predict.py path/to/image.jpg
    python src/predict.py img1.png img2.tif --top 3
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from config import CLASSES, RESULTS_DIR, IMG_SIZE
from model import build_model, get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(model_path=None):
    path = model_path or (RESULTS_DIR / "best_model.pt")
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}. Train first with: python src/main.py")
    device = get_device()
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_images(image_paths, top_k=3, model_path=None):
    model, device = load_model(model_path)
    results = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu()
        topk_probs, topk_indices = probs.topk(top_k)

        result = {
            "file": str(path),
            "top_predictions": [
                {"class": CLASSES[idx], "confidence": float(prob)}
                for idx, prob in zip(topk_indices.tolist(), topk_probs.tolist())
            ],
        }
        results.append(result)
        print(f"\nFile: {path}")
        for i, pred in enumerate(result["top_predictions"], 1):
            bar = "█" * int(pred["confidence"] * 30)
            print(f"  {i}. {pred['class']:<25} {pred['confidence']*100:5.1f}%  {bar}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict land cover class of satellite images")
    parser.add_argument("images", nargs="+", type=Path, help="Path(s) to image file(s)")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions (default: 3)")
    parser.add_argument("--model", type=Path, default=None, help="Path to model checkpoint")
    args = parser.parse_args()

    predict_images(args.images, top_k=args.top, model_path=args.model)


if __name__ == "__main__":
    main()

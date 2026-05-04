import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms
from config import CLASSES, RESULTS_DIR

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor):
    return torch.clamp(tensor * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None], 0, 1)


def plot_class_samples(loader, n_per_class=4, save_path=None):
    buckets = {i: [] for i in range(len(CLASSES))}
    for images, labels in loader:
        for img, lbl in zip(images, labels):
            c = int(lbl)
            if len(buckets[c]) < n_per_class:
                buckets[c].append(denormalize(img).permute(1, 2, 0).numpy())
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    rows, cols = len(CLASSES), n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for r, cls in enumerate(CLASSES):
        for c in range(n_per_class):
            ax = axes[r][c]
            if c < len(buckets[r]):
                ax.imshow(buckets[r][c])
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=0, labelpad=70, va="center", fontweight="bold")

    plt.suptitle("EuroSAT — Sample Images per Class (Sentinel-2 RGB)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or (RESULTS_DIR / "class_samples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class samples saved → {path}")


def plot_predictions(model, loader, device, n=20, save_path=None):
    model.eval()
    images_shown, labels_shown, preds_shown = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu()
            for img, lbl, pred in zip(images, labels, preds):
                images_shown.append(denormalize(img).permute(1, 2, 0).numpy())
                labels_shown.append(int(lbl))
                preds_shown.append(int(pred))
                if len(images_shown) >= n:
                    break
            if len(images_shown) >= n:
                break

    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images_shown[i])
        true_cls = CLASSES[labels_shown[i]]
        pred_cls = CLASSES[preds_shown[i]]
        correct = labels_shown[i] == preds_shown[i]
        color = "#27ae60" if correct else "#e74c3c"
        ax.set_title(f"T: {true_cls}\nP: {pred_cls}", fontsize=7.5, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis("off")

    green_patch = mpatches.Patch(color="#27ae60", label="Correct")
    red_patch = mpatches.Patch(color="#e74c3c", label="Incorrect")
    fig.legend(handles=[green_patch, red_patch], loc="lower right", fontsize=11)
    plt.suptitle("EuroSAT — Model Predictions on Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or (RESULTS_DIR / "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Predictions plot saved → {path}")


def plot_class_distribution(loader, split_name="train", save_path=None):
    counts = np.zeros(len(CLASSES), dtype=int)
    for _, labels in loader:
        for lbl in labels:
            counts[int(lbl)] += 1

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CLASSES, counts, color=plt.cm.tab10(np.linspace(0, 1, len(CLASSES))))
    ax.set_title(f"Class Distribution — {split_name} split", fontsize=14, fontweight="bold")
    ax.set_ylabel("Sample count")
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = save_path or (RESULTS_DIR / f"class_distribution_{split_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Class distribution saved → {path}")

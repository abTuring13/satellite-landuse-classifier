import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import CLASSES, RESULTS_DIR
from model import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, save_path=None):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = save_path or (RESULTS_DIR / "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {path}")


def plot_training_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val", markersize=4)
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", label="Val", markersize=4)
    ax2.set_title("Accuracy (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("EfficientNet-B0 on EuroSAT — Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = save_path or (RESULTS_DIR / "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history saved → {path}")


def run_evaluation(model, test_loader, history=None):
    device = get_device()
    model = model.to(device)

    labels, preds, probs = collect_predictions(model, test_loader, device)

    report = classification_report(labels, preds, target_names=CLASSES, digits=4)
    print("\n" + "=" * 60)
    print("TEST SET — CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    test_acc = (labels == preds).mean()
    metrics = {"test_accuracy": float(test_acc)}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    plot_confusion_matrix(labels, preds)
    if history:
        plot_training_history(history)

    return labels, preds, probs

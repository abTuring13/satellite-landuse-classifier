import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np
from config import CLASSES, DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_eurosat(cache_dir=None):
    print("Downloading EuroSAT dataset from HuggingFace Hub...")
    ds = load_dataset("blanchon/EuroSAT_RGB", cache_dir=str(cache_dir or DATA_DIR))
    return ds


def build_dataloaders(ds, pin_memory=True):
    full = ds["train"]
    n = len(full)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )

    train_ds = EuroSATDataset(full.select(list(train_idx)), transform=train_transform)
    val_ds = EuroSATDataset(full.select(list(val_idx)), transform=eval_transform)
    test_ds = EuroSATDataset(full.select(list(test_idx)), transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)

    print(f"Dataset splits — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader

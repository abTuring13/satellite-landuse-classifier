from pathlib import Path

ROOT = Path(__file__).parent.parent

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_CLASSES = len(CLASSES)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

IMG_SIZE = 64
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
WARMUP_EPOCHS = 2

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

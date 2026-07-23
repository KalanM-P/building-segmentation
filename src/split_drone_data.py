import os
import shutil
from pathlib import Path
import random

DRONE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_dataset"
IMAGE_DIR = os.path.join(DRONE_DIR, "images")
MASK_DIR = os.path.join(DRONE_DIR, "masks")

TRAIN_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/images"
TRAIN_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/masks"
VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/images"
VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/masks"

Path(TRAIN_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAIN_MASK_DIR).mkdir(parents=True, exist_ok=True)
Path(VAL_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(VAL_MASK_DIR).mkdir(parents=True, exist_ok=True)

all_files = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')])

# Fixed seed so the split is reproducible
random.seed(42)
random.shuffle(all_files)

split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"Splitting {len(all_files)} images into {len(train_files)} training and {len(val_files)} validation")

for filename in train_files:
    shutil.copy2(os.path.join(IMAGE_DIR, filename), os.path.join(TRAIN_IMAGE_DIR, filename))
    shutil.copy2(os.path.join(MASK_DIR, filename), os.path.join(TRAIN_MASK_DIR, filename))

for filename in val_files:
    shutil.copy2(os.path.join(IMAGE_DIR, filename), os.path.join(VAL_IMAGE_DIR, filename))
    shutil.copy2(os.path.join(MASK_DIR, filename), os.path.join(VAL_MASK_DIR, filename))

print("Done")
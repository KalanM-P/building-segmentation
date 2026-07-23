import os

DRONE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_dataset"
IMAGE_DIR = os.path.join(DRONE_DIR, "images")
MASK_DIR = os.path.join(DRONE_DIR, "masks")

# Filter out .DS_Store and other hidden files
image_files = [f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')]
mask_files = [f for f in os.listdir(MASK_DIR) if not f.startswith('.')]

print(f"There are {len(image_files)} images and {len(mask_files)} masks")
print("They match up" if len(image_files) == len(mask_files) else "They don't match")
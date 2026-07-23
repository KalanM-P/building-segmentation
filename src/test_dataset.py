from dataset import create_dataloaders
import torch

TRAIN_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/images"
TRAIN_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/masks"
VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/images"
VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/masks"

train_loader, val_loader = create_dataloaders(
    train_image_dir=TRAIN_IMAGE_DIR,
    train_mask_dir=TRAIN_MASK_DIR,
    val_image_dir=VAL_IMAGE_DIR,
    val_mask_dir=VAL_MASK_DIR,
    batch_size=4,
    image_size=256,
    num_workers=0 
)

images, masks = next(iter(train_loader))

print(f"Loaded a batch fine — images {images.shape}, masks {masks.shape}, mask values {torch.unique(masks)}.")
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DubaiBuildingDataset(Dataset):
    """
    Dataset for binary building segmentation from Dubai aerial imagery.
    Converts RGB masks to binary.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks don't match"

        # Maps each RGB color in the mask to its class index
        self.color_to_class = {
            (226, 169, 41): 0,   # Water
            (132, 41, 246): 1,   # Land
            (110, 193, 228): 2,  # Road
            (60, 16, 152): 3,    # Building
            (254, 221, 58): 4,   # Vegetation
            (155, 155, 155): 5   # Unlabeled
        }

    def __len__(self):
        return len(self.image_files)

    def _convert_mask_to_binary(self, mask_rgb):
        """Convert an RGB class-color mask into a binary building mask."""
        if len(mask_rgb.shape) == 2:
            mask_rgb = np.stack([mask_rgb, mask_rgb, mask_rgb], axis=-1)

        mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

        for color, class_id in self.color_to_class.items():
            matches = np.all(mask_rgb == color, axis=-1)
            mask[matches] = class_id

        # Building is class 3 - everything else becomes background
        binary_mask = (mask == 3).astype(np.float32)

        return binary_mask

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path))

        binary_mask = self._convert_mask_to_binary(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']

        # Add channel dimension: (H, W) -> (1, H, W)
        binary_mask = binary_mask.unsqueeze(0)

        return image, binary_mask


def get_transforms(image_size=256, is_train=True):
    """Training gets augmentation; validation only gets resize + normalize."""
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def create_dataloaders(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir,
                       batch_size=4, image_size=256, num_workers=2):
    train_dataset = DubaiBuildingDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=get_transforms(image_size, is_train=True)
    )

    val_dataset = DubaiBuildingDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=get_transforms(image_size, is_train=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"{len(train_dataset)} training and {len(val_dataset)} validation images, batch size {batch_size}, resized to {image_size}x{image_size}")

    return train_loader, val_loader
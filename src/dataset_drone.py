import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DroneBuildingDataset(Dataset):
    """
    Dataset for binary building segmentation from drone aerial imagery.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

        assert len(self.image_files) == len(self.mask_files), \
            "Number of images and masks don't match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Masks can be single-channel or RGB depending on export settings
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Mask pixel values are already 0/1, no thresholding needed
        binary_mask = mask.astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']

        # Add channel dimension: (H, W) -> (1, H, W)
        binary_mask = binary_mask.unsqueeze(0)

        return image, binary_mask


def get_transforms(image_size=256, is_train=True):
    """Training gets augmentation and validation only gets resize + normalize."""
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
                       batch_size=4, image_size=256, num_workers=0):
    train_dataset = DroneBuildingDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=get_transforms(image_size, is_train=True)
    )

    val_dataset = DroneBuildingDataset(
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
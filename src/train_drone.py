import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from dataset_drone import create_dataloaders


def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += iou

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})

    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.sigmoid(outputs)
            iou = calculate_iou(preds, masks)

            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})

    return total_loss / len(loader), total_iou / len(loader)


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_iou'], label='Train IoU')
    ax2.plot(history['val_iou'], label='Val IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Training and Validation IoU')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')


def train_model(train_loader, val_loader, num_epochs=40, device='cpu', save_dir='../models_drone'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",  # transfer learning from ImageNet
        in_channels=3,
        classes=1,
        activation=None  # sigmoid applied manually when computing IoU
    )
    model = model.to(device)

    print(f"I'm training a U-Net with a ResNet34 encoder on {device} for {num_epochs} epochs.")

    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    best_iou = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        scheduler.step(val_iou)

        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"Train loss {train_loss:.4f}, IoU {train_iou:.4f} | Val loss {val_loss:.4f}, IoU {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print(f"That's my best result so far — saving it (IoU {best_iou:.4f}).")

    torch.save(model.state_dict(), f"{save_dir}/final_model.pth")
    plot_training_history(history, f"{save_dir}/training_history.png")

    print(f"Training's done. Best validation IoU I hit was {best_iou:.4f}, saved in {save_dir}/.")

    return model, history


if __name__ == "__main__":
    TRAIN_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/images"
    TRAIN_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/masks"
    VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/images"
    VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/masks"

    BATCH_SIZE = 8  
    IMAGE_SIZE = 256
    NUM_EPOCHS = 40
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    SAVE_DIR = '../models_drone'

    train_loader, val_loader = create_dataloaders(
        train_image_dir=TRAIN_IMAGE_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        val_image_dir=VAL_IMAGE_DIR,
        val_mask_dir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0
    )

    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_dir=SAVE_DIR
    )
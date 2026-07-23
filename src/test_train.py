import torch
from torch.optim import Adam
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import create_dataloaders

TRAIN_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/images"
TRAIN_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/masks"
VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/images"
VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/masks"

BATCH_SIZE = 4
IMAGE_SIZE = 256
NUM_EPOCHS = 2  # sanity check
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"{NUM_EPOCHS}-epoch test on {DEVICE}")

train_loader, val_loader = create_dataloaders(
    train_image_dir=TRAIN_IMAGE_DIR,
    train_mask_dir=TRAIN_MASK_DIR,
    val_image_dir=VAL_IMAGE_DIR,
    val_mask_dir=VAL_MASK_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    num_workers=0
)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
model = model.to(DEVICE)

criterion = smp.losses.DiceLoss(mode='binary')
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

print("Test run finished")
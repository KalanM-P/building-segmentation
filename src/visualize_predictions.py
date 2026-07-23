import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/images"
VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/validation/masks"
MODEL_PATH = "../models/best_model.pth"
SAVE_DIR = "../results"
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Maps each RGB color in the mask to its class index
COLOR_TO_CLASS = {
    (226, 169, 41): 0,   # Water
    (132, 41, 246): 1,   # Land
    (110, 193, 228): 2,  # Road
    (60, 16, 152): 3,    # Building
    (254, 221, 58): 4,   # Vegetation
    (155, 155, 155): 5   # Unlabeled
}


def convert_mask_to_binary(mask_rgb):
    """Convert an RGB class-color mask into a binary building mask."""
    if len(mask_rgb.shape) == 2:
        mask_rgb = np.stack([mask_rgb, mask_rgb, mask_rgb], axis=-1)

    mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)

    for color, class_id in COLOR_TO_CLASS.items():
        matches = np.all(mask_rgb == color, axis=-1)
        mask[matches] = class_id

    binary_mask = (mask == 3).astype(np.float32)
    return binary_mask


def preprocess_image(image):
    """Normalize with ImageNet stats and convert to a batched tensor."""
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image


print("Loading the model")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # loading our own trained weights, not ImageNet ones
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model = model.to(DEVICE)
model.eval()

val_images = sorted([f for f in os.listdir(VAL_IMAGE_DIR) if not f.startswith('.')])
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Generating predictions for {len(val_images)} validation images")

num_to_show = min(6, len(val_images))
fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5*num_to_show))

for i in range(num_to_show):
    img_name = val_images[i]

    img_path = os.path.join(VAL_IMAGE_DIR, img_name)
    image = np.array(Image.open(img_path).convert("RGB"))

    mask_path = os.path.join(VAL_MASK_DIR, img_name.replace('.jpg', '.png'))
    mask_rgb = np.array(Image.open(mask_path))
    gt_mask = convert_mask_to_binary(mask_rgb)

    image_tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_binary = (pred > 0.5).astype(np.float32)

    if num_to_show == 1:
        ax_img, ax_gt, ax_pred = axes
    else:
        ax_img, ax_gt, ax_pred = axes[i]

    ax_img.imshow(image)
    ax_img.set_title(f"Input Image: {img_name}")
    ax_img.axis('off')

    ax_gt.imshow(gt_mask, cmap='gray')
    ax_gt.set_title("Ground Truth (Buildings)")
    ax_gt.axis('off')

    ax_pred.imshow(pred_binary, cmap='gray')
    ax_pred.set_title("Prediction (Buildings)")
    ax_pred.axis('off')

    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)

    print(f"{img_name}: IoU {iou:.4f}")

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/predictions_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved the comparison to {SAVE_DIR}/predictions_comparison.png")
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

VAL_IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/images"
VAL_MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_validation/masks"
MODEL_PATH = "../models_drone/best_model.pth"
SAVE_DIR = "../results"
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def preprocess_image(image):
    """Normalize with ImageNet stats and convert to a batched tensor."""
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return image


print("Loading the drone model")
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

    mask_path = os.path.join(VAL_MASK_DIR, img_name)
    gt_mask = np.array(Image.open(mask_path))
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask[:, :, 0]
    gt_mask = gt_mask.astype(np.float32)

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
    ax_gt.set_title("Ground Truth")
    ax_gt.axis('off')

    ax_pred.imshow(pred_binary, cmap='gray')
    ax_pred.set_title("Prediction")
    ax_pred.axis('off')

    intersection = (pred_binary * gt_mask).sum()
    union = pred_binary.sum() + gt_mask.sum() - intersection
    iou = intersection / (union + 1e-6)

    print(f"{img_name}: IoU {iou:.4f}")

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/drone_predictions_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved the comparison to {SAVE_DIR}/drone_predictions_comparison.png")

# Second visualization: predictions overlaid directly on the source image
fig2, axes2 = plt.subplots(num_to_show, 2, figsize=(12, 6*num_to_show))

for i in range(num_to_show):
    img_name = val_images[i]

    img_path = os.path.join(VAL_IMAGE_DIR, img_name)
    image = np.array(Image.open(img_path).convert("RGB"))

    mask_path = os.path.join(VAL_MASK_DIR, img_name)
    gt_mask = np.array(Image.open(mask_path))
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask[:, :, 0]

    image_tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_binary = (pred > 0.5).astype(np.float32)

    gt_overlay = image.copy()
    pred_overlay = image.copy()

    # Blend building pixels with a solid color at 50% opacity
    gt_overlay[gt_mask > 0.5] = gt_overlay[gt_mask > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
    pred_overlay[pred_binary > 0.5] = pred_overlay[pred_binary > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5

    if num_to_show == 1:
        ax_gt, ax_pred = axes2
    else:
        ax_gt, ax_pred = axes2[i]

    ax_gt.imshow(gt_overlay.astype(np.uint8))
    ax_gt.set_title("Ground Truth Overlay (Red)")
    ax_gt.axis('off')

    ax_pred.imshow(pred_overlay.astype(np.uint8))
    ax_pred.set_title("Prediction Overlay (Green)")
    ax_pred.axis('off')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/drone_predictions_overlay.png', dpi=150, bbox_inches='tight')
print(f"Saved the overlay to {SAVE_DIR}/drone_predictions_overlay.png")
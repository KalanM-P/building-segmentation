import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/images"
MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/drone_training/masks"

files = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')])
first_file = files[0]

img_path = os.path.join(IMAGE_DIR, first_file)
image = np.array(Image.open(img_path).convert("RGB"))

mask_path = os.path.join(MASK_DIR, first_file)
mask = np.array(Image.open(mask_path))

print(f"I'm checking {first_file}. Mask shape is {mask.shape}, dtype {mask.dtype}, values from {mask.min()} to {mask.max()}.")

if len(mask.shape) == 3:
    test_mask = mask[:, :, 0]
else:
    test_mask = mask

# Threshold assumes 0-255 range - masks stored as 0/1 will produce all zeros here
binary = (test_mask > 127).astype(np.float32)
print(f"After thresholding at 127, {binary.mean()*100:.2f}% of pixels come out as buildings.")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Image")
axes[0].axis('off')

if len(mask.shape) == 3:
    axes[1].imshow(mask)
    axes[1].set_title("Mask (RGB)")
else:
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask (Grayscale)")
axes[1].axis('off')

axes[2].imshow(binary, cmap='gray')
axes[2].set_title(f"Binary (>127)\n{binary.mean()*100:.1f}% buildings")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('../results/mask_debug.png', dpi=150)
print("Saved")
plt.show()
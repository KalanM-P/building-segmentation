import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/images"
MASK_DIR = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/data/training/masks"

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if not f.startswith('.')])

print(f"I found {len(image_files)} images and {len(mask_files)} masks.")

img_path = os.path.join(IMAGE_DIR, image_files[0])
mask_path = os.path.join(MASK_DIR, mask_files[0])

image = np.array(Image.open(img_path))
mask_rgb = np.array(Image.open(mask_path))

# Maps each RGB color in the mask to its class index
color_to_class = {
    (226, 169, 41): 0,   # Water
    (132, 41, 246): 1,   # Land
    (110, 193, 228): 2,  # Road
    (60, 16, 152): 3,    # Building
    (254, 221, 58): 4,   # Vegetation
    (155, 155, 155): 5   # Unlabeled
}

mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
for color, class_id in color_to_class.items():
    matches = np.all(mask_rgb == color, axis=-1)
    mask[matches] = class_id

# Building is class 3 - everything else becomes background
binary_mask = (mask == 3).astype(np.uint8)
print(f"Buildings make up {100 * binary_mask.mean():.2f}% of pixels in this image.")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Satellite Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='tab10', vmin=0, vmax=5)
plt.title("Multi-class Mask\n(0=Building, 1=Land, 2=Road, 3=Veg, 4=Water, 5=Unlabeled)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask\n(White=Building, Black=Not Building)")
plt.axis('off')

plt.tight_layout()
results_dir = "/Users/kalanm-p/Documents/Personal/Projects/Building Segmentation/results"
os.makedirs(results_dir, exist_ok=True)
plt.savefig(f'{results_dir}/data_exploration.png', dpi=150, bbox_inches='tight')
print(f"Saved to {results_dir}/data_exploration.png")
plt.show()
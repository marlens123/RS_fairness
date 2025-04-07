import os
import random
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(10)

root = Path("src/data/loveda/All/")
urban_images = list((root / "Urban/images_png").glob("*.png"))
rural_images = list((root / "Rural/images_png").glob("*.png"))

urban_samples = [(str(img), str(img).replace("images_png", "masks_png")) for img in urban_images]
rural_samples = [(str(img), str(img).replace("images_png", "masks_png")) for img in rural_images]

# Combine and create labels
all_samples = urban_samples + rural_samples
labels = ["Urban"] * len(urban_samples) + ["Rural"] * len(rural_samples)

# Stratified split
train_samples, test_samples = train_test_split(all_samples, test_size=0.2, stratify=labels, random_state=10)

def copy_samples(samples, split):
    for img_path, mask_path in samples:
        label = "Urban" if "Urban" in img_path else "Rural"
        for subfolder, src in zip(["images_png", "masks_png"], [img_path, mask_path]):
            dest_dir = root / split / label / subfolder
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src, dest_dir / Path(src).name)

# Copy to train/ and test/
copy_samples(train_samples, "random_train")
copy_samples(test_samples, "random_test")

print(f"Train samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")

from collections import Counter

train_labels = ["Urban" if "Urban" in img else "Rural" for img, _ in train_samples]
test_labels = ["Urban" if "Urban" in img else "Rural" for img, _ in test_samples]

print("Train label distribution:", Counter(train_labels))
print("Test label distribution:", Counter(test_labels))
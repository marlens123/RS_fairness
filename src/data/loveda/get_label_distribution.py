import os
import numpy as np
from PIL import Image
from collections import defaultdict
from collections import Counter

subsets = {
    "rural_train": os.path.join("./Train", "Rural", "masks_png"),
    "urban_train": os.path.join("./Train", "Urban", "masks_png"),
    "rural_test": os.path.join("./Val", "Rural", "masks_png"),
    "urban_test": os.path.join("./Val", "Urban", "masks_png"),
}

class_pixel_counts = defaultdict(Counter)

for subset_name, path in subsets.items():
    print(f"Processing {subset_name}...")
    for file in os.listdir(path):
        if file.endswith(".png"):
            mask = np.array(Image.open(os.path.join(path, file)))
            flat_mask = mask.flatten()
            class_pixel_counts[subset_name].update(flat_mask)

# Print results
for subset, counter in class_pixel_counts.items():
    print(f"\n{subset} class distribution (pixel counts):")
    for cls, count in sorted(counter.items()):
        print(f"  Class {cls}: {count} pixels")

import os
import numpy as np
from PIL import Image
from collections import defaultdict

subsets = {
    "rural_train": os.path.join("./Train", "Rural", "masks_png"),
    "urban_train": os.path.join("./Train", "Urban", "masks_png"),
    "rural_test": os.path.join("./Val", "Rural", "masks_png"),
    "urban_test": os.path.join("./Val", "Urban", "masks_png"),
}

class_counts = {}
unique_classes_per_subset = defaultdict(set)

for subset_name, path in subsets.items():
    print(f"Processing {subset_name}...")
    for file in os.listdir(path):
        if file.endswith(".png"):
            mask = np.array(Image.open(os.path.join(path, file)))
            unique_classes_per_subset[subset_name].update(np.unique(mask))

# Count classes
for subset, class_set in unique_classes_per_subset.items():
    class_counts[subset] = len(class_set)
    print(f"{subset}: {sorted(class_set)} (Total classes: {len(class_set)})")

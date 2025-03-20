# credit the authors
import numpy as np
import io
import os
import torch
import zipfile
import requests
import torch.nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import argparse
from .utils.loveda_dataset import LoveDALoader
from torchmetrics import JaccardIndex
import importlib.util

import satlaspretrain_models

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="src/configs/loveda/loveda.py")

args = argparser.parse_args()

def load_config(config_path):
    """Dynamically load a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# Load the config dynamically
config = load_config(args.config_file)

TRAIN_DATA_CONFIG = config.data["train"]["params"]
VAL_DATA_CONFIG = config.data["test"]["params"]
OPTIMIZER = config.optimizer
LEARNING_RATE = config.learning_rate
NUM_EPOCHS = config.train["num_epochs"]

# Experiment arguments.
device = torch.device('cpu')
val_step = 1  # evaluate every val_step epochs

save_path = 'weights/'  # where to save model weights
os.makedirs(save_path, exist_ok=True)

train_dataloader = LoveDALoader(TRAIN_DATA_CONFIG)
val_dataloader = LoveDALoader(VAL_DATA_CONFIG)

# Initialize a pretrained model, using the SatlasPretrain single-image Swin-v2-Base Sentinel-2 image model weights
# with a segmentation head with num_categories=7, since LoveDA has 7 classes.
weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=satlaspretrain_models.Head.SEGMENT, 
                                                num_categories=TRAIN_DATA_CONFIG["num_classes"], device='cpu')
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

valid = 0
invalid = 0
for data, target in train_dataloader:
    if target['cls'].any() == -1:
        invalid += 1
    else:
        valid += 1

print("Invalid data = ", invalid)
print("Valid data = ", valid)

# Training loop.
for epoch in range(NUM_EPOCHS):
    print("Starting Epoch...", epoch)

    valid_batches = 0

    for data, target in train_dataloader:
        if (target['cls'] == -1).any():
            continue

        valid_batches += 1

        # loss is going to be cross entropy loss per default
        output, loss = model(data, target['cls'])
        print("Train Loss = ", loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        break

    print(f"Epoch {epoch}, valid batches: {valid_batches}")

    # Validation.
    if epoch % val_step == 0:
        model.eval()

        for val_data, val_target in val_dataloader:
            if (target['cls'] == -1).any():
                continue
            val_data = val_data.to(device)
            val_target = val_target.to(device)

            val_output, val_loss = model(val_data, val_target)

            jaccard = JaccardIndex(
                task="multiclass", num_classes=TRAIN_DATA_CONFIG["num_classes"], average="macro"
            ).to(device)
            jac_m = jaccard(val_output, val_target.squeeze(1))

            print("Validation mean IoU = ", jac_m)

            # Comparison IoU computation
            val_labels = torch.argmax(val_output, dim=1)  # Shape: [batch_size, H, W]

            iou_per_class = []
            for cls in range(val_labels.shape[1]):  # Loop over classes
                inter = ((val_labels == cls) & (val_target == cls)).sum()
                union = ((val_labels == cls) | (val_target == cls)).sum()
                if union > 0:
                    iou_per_class.append((inter / union).item())
            mean_iou = np.mean(iou_per_class) if iou_per_class else 0

            print("Compared validation mean IoU = ", mean_iou)

        # Save the model checkpoint at the end of each epoch.
        torch.save(model.state_dict(), save_path + str(epoch) + '_model_weights.pth')


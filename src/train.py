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
import wandb

import sys

#from satlaspretrain_models.satlaspretrain_models.model import Weights, Model
#from satlaspretrain_models.satlaspretrain_models.utils import Head, Backbone

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "satlaspretrain_models")))

from satlaspretrain_models.model import Weights, Model
from satlaspretrain_models.utils import Head, Backbone

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="src/configs/loveda/loveda.py")
argparser.add_argument("--disable_wandb", action="store_true", help="Disable wandb for logging")
argparser.add_argument("--run_name", type=str, default="loveda_test")

args = argparser.parse_args()

def load_config(config_path):
    """Dynamically load a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# Load the config dynamically
config = load_config(args.config_file)

if not args.disable_wandb:
    wandb.login()
    wandb.init(
        entity='sea-ice',
        project='rs_fairness',
        name=args.run_name,
    )

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
weights_manager = Weights()
model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=Head.SEGMENT, 
                                                num_categories=TRAIN_DATA_CONFIG["num_classes"], device='cpu')
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


####################################
# Remove this block later, only for testing
####################################
valid = 0
invalid = 0
for data, target in train_dataloader:
    if target['cls'].any() == -1:
        invalid += 1
    else:
        valid += 1

print("Invalid data = ", invalid)
print("Valid data = ", valid)
####################################
# End of block
####################################


# Training loop.
for epoch in range(NUM_EPOCHS):
    print("Starting Epoch...", epoch)

    train_loss = 0

    for data, target in train_dataloader:
        # loss is going to be cross entropy loss and pixels with -1 are ignored by the loss function
        output, loss = model(data, target['cls'])
        print("Train Loss = ", loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    if not args.disable_wandb:
        wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_dataloader)})

    # Validation.
    if epoch % val_step == 0:
        model.eval()

        val_loss = 0
        jac_m = 0
        mean_iou = 0

        for val_data, val_target in val_dataloader:
            val_target = val_target['cls'].to(device)
            val_output, loss = model(val_data, val_target)

            val_loss += loss.item()

            ####################################################################################
            # Only for testing, remove later
            ####################################################################################
            jaccard = JaccardIndex(
                task="multiclass", num_classes=TRAIN_DATA_CONFIG["num_classes"], average="macro"
            ).to(device)
            mean_jaccard = jaccard(val_output, val_target.squeeze(1))

            jac_m += mean_jaccard.item()

            print("Validation mean IoU = ", mean_jaccard.item())
            ####################################################################################
            # End of testing block
            ####################################################################################

            # Comparison IoU computation
            val_labels = torch.argmax(val_output, dim=1)  # Shape: [batch_size, H, W]

            iou_per_class = []
            for cls in range(val_labels.shape[1]):  # Loop over classes
                inter = ((val_labels == cls) & (val_target == cls)).sum()
                union = ((val_labels == cls) | (val_target == cls)).sum()
                if union > 0:
                    iou_per_class.append((inter / union).item())
            iou_mean = np.mean(iou_per_class) if iou_per_class else 0

            mean_iou += iou_mean

            print("Compared validation mean IoU = ", iou_mean)

        if not args.disable_wandb:
            wandb.log({"epoch": epoch, "val_loss": val_loss/len(val_dataloader)})
            wandb.log({"epoch": epoch, "val_jac": jac_m /len(val_dataloader)})
            wandb.log({"epoch": epoch, "val_miou": mean_iou / len(val_dataloader)})

        # Save the model checkpoint at the end of each epoch.
        torch.save(model.state_dict(), save_path + str(epoch) + '_model_weights.pth')


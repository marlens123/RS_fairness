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

from .satlaspretrain_models.satlaspretrain_models.model import Weights as SatlasWeights

from .imagenetpretrain_models.model import ImageNetWeights

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="src/configs/loveda/adamw_lr0.001.py")
argparser.add_argument("--disable_wandb", action="store_true", help="Disable wandb for logging")
argparser.add_argument("--pretraining_dataset", type=str, default="Satlas", choices=["Satlas", "ImageNet"])
argparser.add_argument("--imagenet_model_identifier", type=str, default="swinb", choices=["swinb", "swint", "resnet50"])
argparser.add_argument("--satlas_model_identifier", type=str, default="Sentinel2_SwinB_SI_RGB", choices=["Aerial_SwinB_SI", "Aerial_SwinB_MI", "Sentinel2_SwinB_SI_RGB", "Sentinel2_SwinB_MI_RGB", "Sentinel2_SwinT_SI_RGB", "Sentinel2_SwinT_MI_RGB", "Sentinel2_Resnet50_SI_RGB", "Sentinel2_Resnet50_MI_RGB"])

args = argparser.parse_args()

# save name should be a combination of the model identifier and the config file name
if args.pretraining_dataset == "Satlas":
    args.run_name = args.pretraining_dataset + "_" + args.satlas_model_identifier + "_" + os.path.basename(args.config_file).split(".")[0].split("/")[-1]
elif args.pretraining_dataset == "ImageNet":
    args.run_name = args.pretraining_dataset + "_" + args.imagenet_model_identifier + "_" + os.path.basename(args.config_file).split(".")[0].split("/")[-1]

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
LEARNING_RATE = config.lr
WEIGHT_DECAY = config.weight_decay
NUM_EPOCHS = config.train["num_epochs"]

# Experiment arguments.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_step = 1  # evaluate every val_step epochs

save_path = 'weights/'  # where to save model weights
os.makedirs(save_path, exist_ok=True)

train_dataloader = LoveDALoader(TRAIN_DATA_CONFIG)
val_dataloader = LoveDALoader(VAL_DATA_CONFIG)

if args.pretraining_dataset == "Satlas":
    from .satlaspretrain_models.satlaspretrain_models.utils import Head
    # load model weights from satlas
    weights_manager = SatlasWeights()
    model = weights_manager.get_pretrained_model(args.satlas_model_identifier, fpn=True, head=Head.SEGMENT, 
                                                    num_categories=TRAIN_DATA_CONFIG["num_classes"], device='cpu')
elif args.pretraining_dataset == "ImageNet":
    from .imagenetpretrain_models.utils import Head
    # load model weights from imagenet
    weights_manager = ImageNetWeights()
    model = weights_manager.get_pretrained_model(backbone=args.imagenet_model_identifier, fpn=True, head=Head.SEGMENT, 
                                                    num_categories=TRAIN_DATA_CONFIG["num_classes"], device='cpu')
else:
    raise ValueError("Invalid pretraining dataset. Choose either 'Satlas' or 'ImageNet'.")

with torch.no_grad():
    weights = model.backbone.features[0][0].weight
    print("Weight stats (mean/std):", weights.mean().item(), weights.std().item())

model = model.to(device)

if config.optimizer == 'adamw':
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif config.optimizer == 'sgd':
    OPTIMIZER = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

# time to iterate through the trainloader
import time
start = time.time()
for data, target in train_dataloader:
    pass
end = time.time()
print("Time to iterate through the trainloader: ", end-start)

scaler = torch.cuda.amp.GradScaler()

# Training loop.
for epoch in range(NUM_EPOCHS):
    print("Starting Epoch...", epoch)

    train_loss = 0

    for data, target in train_dataloader:
        with torch.cuda.amp.autocast():  # Mixed precision
            data, target = data.to(device), target['cls'].to(device)
            # loss is going to be cross entropy loss and pixels with -1 are ignored by the loss function
            output, loss = model(data, target)
            print(f"Train Loss = {loss}", flush=True)
        OPTIMIZER.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update()

        train_loss += loss.item()

    if not args.disable_wandb:
        wandb.log({"epoch": epoch, "train_loss": train_loss/len(train_dataloader)})

    # Validation.
    if epoch % val_step == 0:
        model.eval()

        val_loss = 0
        jac_m = 0
        mean_iou = 0

        with torch.no_grad():
            for val_data, val_target in val_dataloader:
                val_data = val_data.to(device)
                val_target = val_target['cls'].to(device)
                val_output, loss = model(val_data, val_target)

                val_loss += loss.item()

                ####################################################################################
                # Only for testing, remove later
                ####################################################################################
                jaccard = JaccardIndex(
                    task="multiclass", num_classes=TRAIN_DATA_CONFIG["num_classes"], average="macro", ignore_index=-1
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
                for cls in range(val_output.shape[1]):  # Loop over classes
                    valid_mask = val_target != -1
                    inter = ((val_labels == cls) & (val_target == cls) & valid_mask).sum()
                    union = ((val_labels == cls) | (val_target == cls) & valid_mask).sum()
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
        path_to_save = os.path.join(save_path, f'{args.run_name}' + f'{str(epoch)}_model_weights.pth')
        torch.save(model.state_dict(), path_to_save)


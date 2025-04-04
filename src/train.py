# credit the authors
import numpy as np
import os
import torch
import random
import torch.nn
import argparse
from .utils.loveda_dataset import LoveDALoader
import importlib.util
import wandb

from .satlaspretrain_models.satlaspretrain_models.model import Weights as SatlasWeights

from .imagenetpretrain_models.model import ImageNetWeights

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="adamw_lr0.001.py")
argparser.add_argument(
    "--disable_wandb", action="store_true", help="Disable wandb for logging"
)
argparser.add_argument(
    "--pretraining_dataset",
    type=str,
    default="Satlas",
    choices=["Satlas", "ImageNet", "none"],
)
argparser.add_argument(
    "--imagenet_model_identifier",
    type=str,
    default="swinb",
    choices=["swinb", "swint", "resnet50"],
)
argparser.add_argument(
    "--satlas_model_identifier",
    type=str,
    default="Sentinel2_SwinB_SI_RGB",
    choices=[
        "Aerial_SwinB_SI",
        "Aerial_SwinB_MI",
        "Sentinel2_SwinB_SI_RGB",
        "Sentinel2_SwinB_MI_RGB",
        "Sentinel2_SwinT_SI_RGB",
        "Sentinel2_SwinT_MI_RGB",
        "Sentinel2_Resnet50_SI_RGB",
        "Sentinel2_Resnet50_MI_RGB",
    ],
)
argparser.add_argument(
    "--split",
    type=str,
    choices=["random", "geogrpahic", "rural_urban", "urban_rural"],
    default="geographic",
    help="Split to use for training and validation.",
)
argparser.add_argument(
    "--random_seed", type=int, default=5, help="Random seed for reproducibility."
)

args = argparser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
if torch.cuda.is_available():
    print("Setting seed for GPU")
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# save name should be a combination of the model identifier and the config file name
if args.pretraining_dataset == "Satlas":
    args.run_name = (
        args.pretraining_dataset
        + "_"
        + args.satlas_model_identifier
        + "_"
        + os.path.basename(args.config_file).split(".")[0].split("/")[-1]
    )
elif args.pretraining_dataset == "ImageNet":
    args.run_name = (
        args.pretraining_dataset
        + "_"
        + args.imagenet_model_identifier
        + "_"
        + os.path.basename(args.config_file).split(".")[0].split("/")[-1]
    )
elif args.pretraining_dataset == "none":
    args.run_name = (
        "scratch_" + os.path.basename(args.config_file).split(".")[0].split("/")[-1]
    )
else:
    raise ValueError(
        "Invalid pretraining dataset. Choose either 'Satlas', 'ImageNet', or 'none'."
    )


def load_config(config_path):
    """Dynamically load a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


# Load the config dynamically
args.config_file = os.path.join("src/configs/loveda", args.split, args.config_file)

config = load_config(args.config_file)

if not args.disable_wandb:
    wandb.login()
    wandb.init(
        entity="sea-ice",
        project="rs_fairness",
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

save_path = "weights/"  # where to save model weights
os.makedirs(save_path, exist_ok=True)

train_dataloader = LoveDALoader(TRAIN_DATA_CONFIG)
val_dataloader = LoveDALoader(VAL_DATA_CONFIG)

if args.pretraining_dataset == "Satlas":
    from .satlaspretrain_models.satlaspretrain_models.utils import Head

    # load model weights from satlas
    weights_manager = SatlasWeights()
    model = weights_manager.get_pretrained_model(
        args.satlas_model_identifier,
        fpn=True,
        head=Head.SEGMENT,
        num_categories=TRAIN_DATA_CONFIG["num_classes"],
        device="cpu",
    )
elif args.pretraining_dataset == "ImageNet":
    from .imagenetpretrain_models.utils import Head

    # load model weights from imagenet
    weights_manager = ImageNetWeights()
    model = weights_manager.get_pretrained_model(
        backbone=args.imagenet_model_identifier,
        fpn=True,
        head=Head.SEGMENT,
        num_categories=TRAIN_DATA_CONFIG["num_classes"],
        device="cpu",
    )
elif args.pretraining_dataset == "none":
    from .imagenetpretrain_models.utils import Head

    # load model weights from imagenet
    weights_manager = ImageNetWeights()
    model = weights_manager.get_pretrained_model(
        backbone=args.imagenet_model_identifier,
        fpn=True,
        head=Head.SEGMENT,
        num_categories=TRAIN_DATA_CONFIG["num_classes"],
        device="cpu",
        weights=None,
    )
else:
    raise ValueError(
        "Invalid pretraining dataset. Choose either 'Satlas' or 'ImageNet'."
    )

model = model.to(device)

if config.optimizer == "adamw":
    OPTIMIZER = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
elif config.optimizer == "sgd":
    OPTIMIZER = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

# time to iterate through the trainloader
import time

start = time.time()
for data, target in train_dataloader:
    pass
end = time.time()
print("Time to iterate through the trainloader: ", end - start)

scaler = torch.cuda.amp.GradScaler()

# Training loop.
for epoch in range(NUM_EPOCHS):
    print("Starting Epoch...", epoch)

    train_loss = 0

    for data, target in train_dataloader:
        with torch.cuda.amp.autocast():  # Mixed precision
            data, target = data.to(device), target["cls"].to(device)
            # loss is going to be cross entropy loss and pixels with -1 are ignored by the loss function
            output, loss = model(data, target)
            print(f"Train Loss = {loss}", flush=True)
        OPTIMIZER.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(OPTIMIZER)
        scaler.update()

        train_loss += loss.item()

    if not args.disable_wandb:
        wandb.log({"epoch": epoch, "train_loss": train_loss / len(train_dataloader)})

    # Validation.
    if epoch % val_step == 0:
        model.eval()

        val_loss = 0
        jac_m = 0
        mean_iou = 0
        per_class_iou = np.zeros((VAL_DATA_CONFIG["num_classes"],))

        with torch.no_grad():
            for val_data, val_target in val_dataloader:
                val_data = val_data.to(device)
                val_target = val_target["cls"].to(device)
                val_output, loss = model(val_data, val_target)

                val_loss += loss.item()

                # Comparison IoU computation
                val_labels = torch.argmax(
                    val_output, dim=1
                )  # Shape: [batch_size, H, W]

                iou_per_class = []
                for cls in range(val_output.shape[1]):  # Loop over classes
                    valid_mask = val_target != -1
                    inter = (
                        (val_labels == cls) & (val_target == cls) & valid_mask
                    ).sum()
                    union = (
                        (val_labels == cls) | (val_target == cls) & valid_mask
                    ).sum()
                    if union > 0:
                        iou_per_class.append((inter / union).item())
                iou_mean = np.mean(iou_per_class) if iou_per_class else 0

                mean_iou += iou_mean
                per_class_iou += np.array(iou_per_class)

                print("Compared validation mean IoU = ", iou_mean)

        # report performance and fairness metrics
        if not args.disable_wandb:
            wandb.log({"epoch": epoch, "val_loss": val_loss / len(val_dataloader)})
            wandb.log({"epoch": epoch, "val_miou": mean_iou / len(val_dataloader)})

            for cls in range(VAL_DATA_CONFIG["num_classes"]):
                wandb.log(
                    {f"val_iou_class_{cls}": per_class_iou[cls] / len(val_dataloader)}
                )

            # report standard deviation of the per class iou
            wandb.log({"class_std": np.std(per_class_iou / len(val_dataloader))})

            # worst class iou
            worst_class_iou = np.min(per_class_iou / len(val_dataloader))
            wandb.log({"worst_class_iou": worst_class_iou})

            # mean of the bottom 30% of the classes
            bottom_30_percent_classes = np.argsort(per_class_iou / len(val_dataloader))[
                : int(0.3 * VAL_DATA_CONFIG["num_classes"])
            ]
            mean_bottom_30_percent_classes = np.mean(
                per_class_iou[bottom_30_percent_classes] / len(val_dataloader)
            )
            wandb.log(
                {"mean_bottom_30_percent_classes": mean_bottom_30_percent_classes}
            )

            # mean of the top 30% of the classes
            top_30_percent_classes = np.argsort(per_class_iou / len(val_dataloader))[
                -int(0.3 * VAL_DATA_CONFIG["num_classes"]) :
            ]
            mean_top_30_percent_classes = np.mean(
                per_class_iou[top_30_percent_classes] / len(val_dataloader)
            )
            wandb.log({"mean_top_30_percent_classes": mean_top_30_percent_classes})

        # Save the model checkpoint at the end of each epoch.
        path_to_save = os.path.join(
            save_path, f"{args.run_name}" + f"{str(epoch)}_model_weights.pth"
        )
        torch.save(model.state_dict(), path_to_save)

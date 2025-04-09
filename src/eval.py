import torch
import argparse
import os
import numpy as np
from .utils.loveda_dataset import LoveDALoader
import importlib.util
import wandb
import random

from .satlaspretrain_models.satlaspretrain_models.model import Weights as SatlasWeights

from .imagenetpretrain_models.model import ImageNetWeights

argparser = argparse.ArgumentParser()
argparser.add_argument("--saved_weights", type=str, default="adamw_lr0.0005.py")
argparser.add_argument("--config_file", type=str, default="adamw_lr0.0005.py")
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
    choices=["random", "geographic", "rural_urban", "urban_rural"],
    default="geographic",
    help="Split to use for training and validation.",
)
argparser.add_argument(
    "--random_seed", type=int, default=10, help="Random seed for reproducibility."
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

model_save_path = os.path.join(f"final_weights/{args.saved_weights}")

def load_config(config_path):
    """Dynamically load a Python module from a given file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module

# Load the config dynamically
args.config_file = os.path.join("src/configs/loveda", args.split, args.config_file)

config = load_config(args.config_file)

wandb.login()
wandb.init(
    entity="sea-ice",
    project="final_fairness",
    name=args.saved_weights,
)

TRAIN_DATA_CONFIG = config.data["train"]["params"]
VAL_DATA_CONFIG_FULL = config.data["test"]["full"]["params"]
VAL_DATA_CONFIG_URBAN = config.data["test"]["urban"]["params"]
VAL_DATA_CONFIG_RURAL = config.data["test"]["rural"]["params"]

train_dataloader = LoveDALoader(TRAIN_DATA_CONFIG)

val_dataloaders = {
    "full": LoveDALoader(VAL_DATA_CONFIG_FULL),
    "urban": LoveDALoader(VAL_DATA_CONFIG_URBAN),
    "rural": LoveDALoader(VAL_DATA_CONFIG_RURAL),
}

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model weights from the saved path
model.load_state_dict(
    torch.load(model_save_path, map_location=torch.device("cpu"))
)

model.to(device)
model.eval()

for id, val_loader in val_dataloaders.items():
    val_loss = 0
    jac_m = 0
    mean_iou = 0
    per_class_iou = np.zeros((VAL_DATA_CONFIG_FULL["num_classes"],))

    with torch.no_grad():
        for idx, (val_data, val_target) in enumerate(val_loader):
            val_data = val_data.to(device)
            val_target = val_target["cls"].to(device)
            val_output, loss = model(val_data, val_target)

            val_loss += loss.item()
            val_labels = torch.argmax(
                val_output, dim=1
            )  # Shape: [batch_size, H, W]

            val_labels = val_labels.cpu().numpy()
            val_target = val_target.cpu().numpy()

            # visualize the output
            import matplotlib.pyplot as plt

            if idx == 1:
                # min max scaling for visualization
                val_labels_vis = val_labels[0]
                val_labels_vis = (val_labels - val_labels.min()) / (
                    val_labels.max() - val_labels.min()
                )
                val_labels_vis = (val_labels_vis * 255).astype(np.uint8)
                val_labels_vis = np.moveaxis(val_labels_vis, 0, -1)
                val_labels_vis = np.repeat(val_labels_vis[:, :, np.newaxis], 3, axis=2)
                val_labels_vis = np.concatenate(
                    [val_labels_vis, val_labels_vis, val_labels_vis], axis=2
                )
                val_labels_vis = np.clip(val_labels_vis, 0, 255).astype(np.uint8)
                # do the same for val_data
                val_data_vis = val_data[0].cpu().numpy().transpose(1, 2, 0)
                val_data_vis = (val_data_vis - val_data_vis.min()) / (
                    val_data_vis.max() - val_data_vis.min()
                )
                val_data_vis = (val_data_vis * 255).astype(np.uint8)
                val_data_vis = np.clip(val_data_vis, 0, 255).astype(np.uint8)
                val_data_vis = np.repeat(val_data_vis[:, :, np.newaxis], 3, axis=2)
                val_data_vis = np.concatenate(
                    [val_data_vis, val_data_vis, val_data_vis], axis=2
                )
                val_data_vis = np.clip(val_data_vis, 0, 255).astype(np.uint8)
                # do the same for val_target
                val_target_vis = val_target[0]
                val_target_vis = (val_target_vis - val_target_vis.min()) / (
                    val_target_vis.max() - val_target_vis.min()
                )
                val_target_vis = (val_target_vis * 255).astype(np.uint8)
                val_target_vis = np.moveaxis(val_target_vis, 0, -1)
                val_target_vis = np.repeat(val_target_vis[:, :, np.newaxis], 3, axis=2)
                val_target_vis = np.concatenate(
                    [val_target_vis, val_target_vis, val_target_vis], axis=2
                )
                val_target_vis = np.clip(val_target_vis, 0, 255).astype(np.uint8)

                # save the images
                plt.imsave(f"assets/input_{args.split}_{args.saved_weights}_{id}.png", val_data_vis)
                plt.imsave(f"assets/output_{args.split}_{args.saved_weights}_{id}.png", val_labels_vis)
                plt.imsave(f"assets/target_{args.split}_{args.saved_weights}_{id}.png", val_target_vis)

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
                else:
                    iou_per_class.append(0)
            iou_mean = np.mean(iou_per_class) if iou_per_class else 0

            mean_iou += iou_mean
            per_class_iou += np.array(iou_per_class)

    # report performance and fairness metrics
    if not args.disable_wandb:
        wandb.log({f"val_loss_{id}": val_loss / len(val_loader)})
        wandb.log({f"val_miou_{id}": mean_iou / len(val_loader)})

        for cls in range(VAL_DATA_CONFIG_FULL["num_classes"]):
            wandb.log(
                {f"val_iou_class_{cls}_{id}": per_class_iou[cls] / len(val_loader)}
            )

        # report standard deviation of the per class iou
        wandb.log({f"class_std_{id}": np.std(per_class_iou / len(val_loader))})

        # worst class iou
        worst_class_iou = np.min(per_class_iou / len(val_loader))
        wandb.log({f"worst_class_iou_{id}": worst_class_iou})

        # mean of the bottom 30% of the classes
        bottom_30_percent_classes = np.argsort(per_class_iou / len(val_loader))[
            : int(0.3 * VAL_DATA_CONFIG_FULL["num_classes"])
        ]
        mean_bottom_30_percent_classes = np.mean(
            per_class_iou[bottom_30_percent_classes] / len(val_loader)
        )
        wandb.log(
            {f"mean_bottom_30_percent_classes_{id}": mean_bottom_30_percent_classes}
        )

        # mean of the top 30% of the classes
        top_30_percent_classes = np.argsort(per_class_iou / len(val_loader))[
            -int(0.3 * VAL_DATA_CONFIG_FULL["num_classes"]) :
        ]
        mean_top_30_percent_classes = np.mean(
            per_class_iou[top_30_percent_classes] / len(val_loader)
        )
        wandb.log({f"mean_top_30_percent_classes_{id}": mean_top_30_percent_classes})

        # print all metrics
        print(f"val_loss_{id}: {val_loss / len(val_loader)}")
        print(f"val_miou_{id}: {mean_iou / len(val_loader)}")
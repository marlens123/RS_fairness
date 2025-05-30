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
argparser.add_argument(
    "--visualize",
    action="store_true",
    help="Visualize the output of the model on the validation set.",
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

model_save_path = os.path.join(f"{args.split}_weights/{args.saved_weights}")

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
VAL_DATA_CONFIG_VIS = config.data["test"]["vis"]["params"]

train_dataloader = LoveDALoader(TRAIN_DATA_CONFIG)

val_dataloaders = {
    "full": LoveDALoader(VAL_DATA_CONFIG_FULL),
    "urban": LoveDALoader(VAL_DATA_CONFIG_URBAN),
    "rural": LoveDALoader(VAL_DATA_CONFIG_RURAL),
}

# contains one sample image and one sample mask for visualization
vis_val_dataloader = LoveDALoader(VAL_DATA_CONFIG_VIS)

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

if args.visualize:
    with torch.no_grad():
        for idx, (val_data, val_target) in enumerate(vis_val_dataloader):
            val_data = val_data.to(device)
            val_target = val_target["cls"].to(device)
            val_output, loss = model(val_data, val_target)

            val_labels = torch.argmax(
                val_output, dim=1
            )  # Shape: [batch_size, H, W]

            val_labels = val_labels.cpu().numpy()
            val_target = val_target.cpu().numpy()

            if idx == 1:
                # min max scaling for visualization
                val_labels_vis = val_labels[0]
                # do the same for val_data
                val_data_vis = val_data[0].cpu().numpy()
                # do the same for val_target
                val_target_vis = val_target[0]

                # save the images
                np.save(f"assets/visualizations/input_{args.split}_{args.saved_weights}.npy", val_data_vis)
                np.save(f"assets/visualizations/output_{args.split}_{args.saved_weights}.npy", val_labels_vis)
                np.save(f"assets/visualizations/target_{args.split}_{args.saved_weights}.npy", val_target_vis)


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

    # store the per-class iou in a csv file, 
    import pandas as pd

    per_class_iou_df = pd.DataFrame(
        per_class_iou / len(val_loader), columns=["iou"], index=range(VAL_DATA_CONFIG_FULL["num_classes"])
    )
    # add the miou, worst class iou, mean bottom 30% classes, mean top 30% classes to the dataframe
    per_class_iou_df["val_miou"] = mean_iou / len(val_loader)
    per_class_iou_df["worst_class_iou"] = np.min(per_class_iou / len(val_loader))
    per_class_iou_df["mean_bottom_30_percent_classes"] = np.mean(
        per_class_iou[np.argsort(per_class_iou / len(val_loader))[: int(0.3 * VAL_DATA_CONFIG_FULL["num_classes"])]]
        / len(val_loader)
    )
    per_class_iou_df["mean_top_30_percent_classes"] = np.mean(
        per_class_iou[np.argsort(per_class_iou / len(val_loader))[-int(0.3 * VAL_DATA_CONFIG_FULL["num_classes"]):]]
        / len(val_loader)
    )
    per_class_iou_df["class_std"] = np.std(per_class_iou / len(val_loader))

    per_class_iou_df.to_csv(f"assets/per_class_iou_{args.split}_{args.saved_weights}_{id}.csv")

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
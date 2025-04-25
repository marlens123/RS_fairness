from torch import nn
import torch
import torch.nn.functional as F
import argparse
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--source_data",
    type=str,
    default="naip",
    choices=["naip", "sentinel2", "ImageNet"],
)
argparser.add_argument(
    "--target_data",
    type=str,
    choices=["Full", "Urban", "Rural"],
    default="Full",
)
argparser.add_argument(
    "--random_seed", type=int, default=5, help="Random seed for reproducibility."
)

args = argparser.parse_args()

source_path = f"src/data/satlas/{args.source_data}/{args.source_data}_small/all_images/"
target_path = f"src/data/loveda/All/{args.target_data}/images_png/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# a quick dataset class
class CostumImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

source_dataset = CostumImageDataset(source_path, transform=transform)
target_dataset = CostumImageDataset(target_path, transform=transform)

source_loader = DataLoader(source_dataset, batch_size=500, shuffle=False)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=False)

# 4. Extract features
def extract_features(dataloader):
    feats = []
    with torch.no_grad():
        for x in tqdm(dataloader):
            x = x.to(device)
            f = resnet(x)
            feats.append(f.cpu())
    return torch.cat(feats)

features_source = extract_features(source_loader)
features_target = extract_features(target_loader)

def compute_mmd(x, y, sigma=1.0):
    """Unbiased MMD^2 using RBF kernel"""
    m, n = x.size(0), y.size(0)

    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = x.pow(2).sum(1).unsqueeze(1)
    ry = y.pow(2).sum(1).unsqueeze(1)

    dxx = rx + rx.t() - 2 * xx
    dyy = ry + ry.t() - 2 * yy
    dxy = rx + ry.t() - 2 * xy

    kxx = torch.exp(-dxx / (2 * sigma ** 2))
    kyy = torch.exp(-dyy / (2 * sigma ** 2))
    kxy = torch.exp(-dxy / (2 * sigma ** 2))

    return (kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1)) - 2 * kxy.mean()

def subsample(tensor, max_samples=5000):
    if len(tensor) > max_samples:
        indices = torch.randperm(len(tensor))[:max_samples]
        return tensor[indices]
    return tensor

features_source_sub = subsample(features_source, max_samples=5000)
features_target_sub = subsample(features_target, max_samples=5000)

mmd_score = compute_mmd(features_source_sub, features_target_sub)

print(f"MMD (RBF) between {args.source_data} and {args.target_data} feature distributions:", mmd_score.item(), flush=True)


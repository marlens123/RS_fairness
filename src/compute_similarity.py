from torch import nn
import torch
import torch.nn.functional as F
from .utils.mmd import compute_mmd
import argparse
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="adamw_lr0.001.py")
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
    default="full",
)
argparser.add_argument(
    "--random_seed", type=int, default=5, help="Random seed for reproducibility."
)

args = argparser.parse_args()

source_path = f"src/data/satlas/{args.source_data}/{args.source_data}_small/"
target_path = f"src/data/loveda/All/{args.target_data}/images_png/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define feature extractor
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval().to(device)

# 2. Define transformation (consistent for both sets)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

source_dataset = ImageFolder(source_path, transform=transform)
target_dataset = ImageFolder(target_path, transform=transform)

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=False)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=False)

# 4. Extract features
def extract_features(dataloader):
    feats = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.cuda()
            f = resnet(x)
            feats.append(f.cpu())
    return torch.cat(feats)

features_source = extract_features(source_loader)
features_target = extract_features(target_loader)

# 5. Compute MMD
def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """Unbiased MMD^2 (squared) using RBF kernel"""
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy

    kxx = torch.exp(-dxx / (2 * sigma ** 2))
    kyy = torch.exp(-dyy / (2 * sigma ** 2))
    kxy = torch.exp(-dxy / (2 * sigma ** 2))

    m = x.size(0)
    n = y.size(0)

    return (kxx.sum() - m) / (m * (m - 1)) + (kyy.sum() - n) / (n * (n - 1)) - 2 * kxy.mean()

mmd_score = compute_mmd(features_source, features_target)
print("MMD (RBF) between source and target feature distributions:", mmd_score.item())



























optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=1e-3)
NUM_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)
classifier.to(device)

lambda_mmd = 0.5  # weight of MMD loss

for epoch in range(NUM_EPOCHS):
    for (x_src, y_src), (x_tgt, _) in zip(source_loader, target_loader):
        x_src, y_src = x_src.to(device), y_src.to(device)
        x_tgt = x_tgt.to(device)

        # Forward pass
        feat_src = feature_extractor(x_src)
        feat_tgt = feature_extractor(x_tgt)

        preds_src = classifier(feat_src)

        # Losses
        task_loss = F.cross_entropy(preds_src, y_src)
        mmd_loss = compute_mmd(feat_src, feat_tgt)

        total_loss = task_loss + lambda_mmd * mmd_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: task_loss = {task_loss.item():.4f}, mmd_loss = {mmd_loss.item():.4f}")

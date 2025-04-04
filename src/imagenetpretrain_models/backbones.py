"""
Inspired by satlaspretrain_models/models/backbones.py, adjusted to use ImageNet weights.
"""

import torch.nn
import torchvision


class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch, weights="IMAGENET1K_V1"):
        super(SwinBackbone, self).__init__()

        print(f"Using {arch} backbone with {weights} weights.", flush=True)

        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b(weights=weights)
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t(weights=weights)
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]
        else:
            raise ValueError("Backbone architecture not supported.")

        if num_channels != 3:
            print(
                f"Changing input channels from 3 to {num_channels}. Note that this layer won't be pretrained.",
                flush=True,
            )
            self.backbone.features[0][0] = torch.nn.Conv2d(
                num_channels,
                self.backbone.features[0][0].out_channels,
                kernel_size=(4, 4),
                stride=(4, 4),
            )

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch="resnet50", weights="IMAGENET1K_V1"):
        super(ResnetBackbone, self).__init__()

        if arch == "resnet50":
            self.resnet = torchvision.models.resnet.resnet50(weights=weights)
            ch = [256, 512, 1024, 2048]
        elif arch == "resnet152":
            self.resnet = torchvision.models.resnet.resnet152(weights=weights)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone architecture not supported.")

        if num_channels != 3:
            print(
                f"Changing input channels from 3 to {num_channels}. Note that this layer won't be pretrained.",
                flush=True,
            )
            self.resnet.conv1 = torch.nn.Conv2d(
                num_channels,
                self.resnet.conv1.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]

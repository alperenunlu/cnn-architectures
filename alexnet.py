from functools import partial

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors

"""
@inproceedings{NIPS2012_c399862d,
 author = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {F. Pereira and C.J. Burges and L. Bottou and K.Q. Weinberger},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {ImageNet Classification with Deep Convolutional Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf},
 volume = {25},
 year = {2012}
}
"""


# Architecture


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=(48 + 48),
                kernel_size=(11, 11),
                stride=4,
                padding=2,
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
            nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75, k=2),  # ppr 3.3
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # ppr 3.4
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=(48 + 48),
                out_channels=(128 + 128),
                kernel_size=(5, 5),
                stride=1,
                padding=2,
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
            nn.LocalResponseNorm(size=5, alpha=1e-3, beta=0.75, k=2),  # ppr 3.3
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # ppr 3.4
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=(128 + 128),
                out_channels=(192 + 192),
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=(192 + 192),
                out_channels=(192 + 192),
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(
                in_channels=(192 + 192),
                out_channels=(128 + 128),
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),  # ppr 3.4
        )

        self.layer6 = nn.Sequential(
            nn.Linear(
                in_features=(128 + 128) * (6 * 6),
                out_features=(2048 + 2048),
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
            nn.Dropout(dropout),  # ppr 4.2
        )

        self.layer7 = nn.Sequential(
            nn.Linear(
                in_features=(2048 + 2048),
                out_features=(2048 + 2048),
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
            nn.Dropout(dropout),  # ppr 4.2
        )

        self.layer8 = nn.Sequential(
            nn.Linear(
                in_features=(2048 + 2048),
                out_features=(num_classes),
            ),  # ppr 3.5
            nn.ReLU(inplace=True),  # ppr 3.1
        )

        # ppr 5
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.layer2[0].bias)
        nn.init.ones_(self.layer4[0].bias)
        nn.init.ones_(self.layer5[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = F.log_softmax(x, -1)
        return x


# Optimizer and Scheduler

# ppr 5
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LEARNING_RATE = 0.01

AlexNet_optimizer = partial(
    optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
AlexNet_scheduler = optim.lr_scheduler.ReduceLROnPlateau


# Preprocessing and Augmentations

# ppr 4.1
AlexNet_train_transforms = T.Compose(
    [
        T.ToImage(),
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5,
        ),
        T.ToDtype(torch.float32, scale=True),
    ]
)


class BatchMultiCrop(T.Transform):
    def forward(self, crops: Tuple[tv_tensors.Image]):
        crops = tv_tensors.wrap(torch.stack(crops), like=crops[0])
        return crops


AlexNet_test_transforms = T.Compose(
    [
        T.ToImage(),
        T.Resize((256, 256)),
        T.TenCrop(224),
        BatchMultiCrop(),
        T.ToDtype(torch.float32, scale=True),
    ]
)

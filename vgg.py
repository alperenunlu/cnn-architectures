from functools import partial

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import optim

import torchvision.transforms.v2 as T

"""
@misc{simonyan2015deepconvolutionalnetworkslargescale,
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
      author={Karen Simonyan and Andrew Zisserman},
      year={2015},
      eprint={1409.1556},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1409.1556},
}
"""


# Architecture


# ppr 2.1
def make_vgg_layers(config: List[Union[str, int]], batch_norm: Optional[bool] = False) -> nn.Sequential:
    """Create feature extraction layers for VGG architectures.

    This function creates a sequential container of layers based on the VGG architecture configuration.
    It supports adding convolutional layers, max-pooling layers, and optional batch normalization layers.

    NOTE: The original VGG paper doesn't use batch normalization layers.

    Args:
        config (List[Union[str, int]]): Configuration list for VGG architecture where each element is either:
            - int: Number of output channels for a Conv2D layer.
            - str: 'M' indicating a MaxPool2D layer.
        batch_norm (Optional[bool]): If True, BatchNorm2D layers are added after each Conv2D layer. Default is False.

    Returns:
        nn.Sequential: A sequential container of the specified VGG layers.
    """
    layers: List[nn.Module] = []
    in_channels = 3

    for v in config:
        if isinstance(v, int):
            layers.append(
                nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1
                )  # , bias=not batch_norm) # torch implementation convolutional layer has bias
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
        elif isinstance(v, str) and v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            raise ValueError(f"Invalid layer type: {v}. Expected int or 'M'.")

    return nn.Sequential(*layers)


# ppr table 1
cfg: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # "C": [], # removed for simplicity
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        dropout: float = 0.5,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # to ensure 7x7
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        # ppr 3.1
        if init_weights:
            # The authors noted that Glorot initialization could have been used.
            # gain = nn.init.calculate_gain("relu") # glorot
            for m in self.modules():
                # nn.init.xavier_uniform_(m.weight, gain)
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG11(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["A"], False), num_classes, dropout, init_weights)


class VGG11_BN(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["A"], True), num_classes, dropout, init_weights)


class VGG13(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["B"], False), num_classes, dropout, init_weights)


class VGG13_BN(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["B"], True), num_classes, dropout, init_weights)


class VGG16(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["D"], False), num_classes, dropout, init_weights)


class VGG16_BN(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["D"], True), num_classes, dropout, init_weights)


class VGG19(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["E"], False), num_classes, dropout, init_weights)


class VGG19_BN(VGG):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = True) -> None:
        super().__init__(make_vgg_layers(cfg["E"], True), num_classes, dropout, init_weights)


# Optimizer and Scheduler
# ppr 3.1

BATCH_SIZE = 256
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LEARNING_RATE = 0.01

VGG_optimizer = partial(optim.SGD, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

VGG_scheduler = optim.lr_scheduler.ReduceLROnPlateau


# Preprocessing and Augmentations

# ppr 3.1
S = 256
# S = 384

VGG_train_transforms = T.Compose(
    [
        T.ToImage(),
        T.Resize(S),  # ppr 3.1
        T.RandomCrop(224),  # ppr 3.1
        T.RandomHorizontalFlip(),  # ppr 3.1
        T.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5,
        ),  # ppr 3.1
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(((0.485, 0.456, 0.406)), ((0.229, 0.224, 0.225))),  # ppr 2.1
    ]
)

# Multi-Scale Training

# S_min = 256
# S_max = 512

# VGG_train_transforms = T.Compose(
#     [
#         T.ToImage(),
#         T.Resize(int(torch.randint(S_min, S_max, (1,)))),  # ppr 3.1
#         T.RandomCrop(224),  # ppr 3.1
#         T.RandomHorizontalFlip(),  # ppr 3.1
#         T.ColorJitter(
#             brightness=0.5,
#             contrast=0.5,
#             saturation=0.5,
#             hue=0.5,
#         ),  # ppr 3.1
#         T.ToDtype(torch.float32, scale=True),
#         T.Normalize(((0.485, 0.456, 0.406)), ((0.229, 0.224, 0.225))),  # ppr 2.1
#     ]
# )


# ppr 4.1
Q = 256
# For Multi-Scale
# Q = 0.5 * (256 + 512)

VGG_test_transforms1 = T.Compose(
    [
        T.ToImage(),
        T.Resize(Q),  # ppr 3.2
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(((0.485, 0.456, 0.406)), ((0.229, 0.224, 0.225))),  # ppr 2.1
    ]
)

VGG_test_transforms2 = T.Compose(
    [
        T.ToImage(),
        T.Resize(Q),  # ppr 3.2
        T.RandomHorizontalFlip(p=1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(((0.485, 0.456, 0.406)), ((0.229, 0.224, 0.225))),  # ppr 2.1
    ]
)


"""
Testing  # ppr 3.2

1. Convert the first Linear layer to a 7x7 Convolutional layer.
2. Convert the last two Linear layers to 1x1 Convolutional layers.
3. Apply a fully convolutional layer to the uncropped image.
4. Spatially average to obtain the class dimensions.
5. Repeat steps 1-4 for horizontally flipped images.
6. Take the average for each image and its horizontally flipped version.

Example of converting Linear Layers to Convolutional Layers:
```python
num_classes = 1000

fc1 = nn.Linear(512 * 7 * 7, 4096)
fc2 = nn.Linear(4096, 4096)
fc3 = nn.Linear(4096, num_classes)

fc = nn.Sequential(
    fc1,
    nn.ReLU(inplace=True),
    fc2,
    nn.ReLU(inplace=True),
    fc3,
)

conv1 = nn.Conv2d(512, 4096, 7)
conv2 = nn.Conv2d(4096, 4096, 1)
conv3 = nn.Conv2d(4096, num_classes, 1)

conv1.weight = nn.Parameter(fc1.weight.clone().view(fc1.out_features, conv1.in_channels, *conv1.kernel_size))
conv1.bias = nn.Parameter(fc1.bias.clone())

conv2.weight = nn.Parameter(fc2.weight.clone().view(fc2.out_features, conv2.in_channels, *conv2.kernel_size))
conv2.bias = nn.Parameter(fc2.bias.clone())

conv3.weight = nn.Parameter(fc3.weight.clone().view(fc3.out_features, conv3.in_channels, *conv3.kernel_size))
conv3.bias = nn.Parameter(fc3.bias.clone())

conv = nn.Sequential(
    conv1,
    nn.ReLU(inplace=True),
    conv2,
    nn.ReLU(inplace=True),
    conv3,
    nn.AdaptiveAvgPool2d(1)
)

a = torch.randn(1, 512, 7, 7)
torch.allclose(fc(a.flatten(1)).flatten(), conv(a).flatten(), atol=1e-5)
```
"""

from . import Model

import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers, cfgs


class VGG11(Model):
    """
    To verify
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5):
        super(VGG11, self).__init__()
        self.features = make_layers(cfgs["A"])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

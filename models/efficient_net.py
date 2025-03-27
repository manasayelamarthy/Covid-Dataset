import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class efficientnetClassifier(nn.Module):
    def __init__(self,num_classes):
        super(efficientnetClassifier, self).__init__()

        self.efficientnet = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.efficientnet(x)
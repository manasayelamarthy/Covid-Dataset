import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torch.optim import Adam

from config import train_Config


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResnetClassifier, self).__init__()

        # Resnet backbone
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Fully connected layer
        features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
    
class resnetModel:
    def __init__(self, config = train_Config()):
        self.model = ResnetClassifier(  num_classes = config.num_classes)
        self.loss  = config.loss
        self.optimizer = Adam(self.model.parameters(), lr = config.learning_rate)


if __name__ == "__main__":
    model = ResnetClassifier(num_classes = 3)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torch.optim import Adam

from config import train_Config


class efficientnetClassifier(nn.Module):
    def __init__(self,num_classes):
        super(efficientnetClassifier, self).__init__()

        self.input_layer = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 1)
        self.efficientnet = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)

        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        return self.efficientnet(x)
    

class efficientModel:
    def __init__(self, config = train_Config()):
        self.model = efficientnetClassifier(  num_classes = config.num_classes)
        self.loss  = config.loss
        self.optimizer = Adam(self.model.parameters(), lr = config.learning_rate)
    
if __name__ == "__main__":
    model = efficientnetClassifier(num_classes = 3)
from .cnn_model import CNNModel
from .efficient_net import efficientnetClassifier
from .resnet import ResnetClassifier

all_model = {
    'cnn_model' : CNNModel,
    'efficient_net' : efficientnetClassifier,
    'resnet': ResnetClassifier
}
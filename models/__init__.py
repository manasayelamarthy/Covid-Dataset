from .cnn_model import CNNModel, cnnModel
from .efficient_net import efficientnetClassifier,efficientModel
from .resnet import ResnetClassifier,resnetModel

all_models = {
    'cnn_model' : cnnModel,
    'efficient_net' :  efficientModel,
    'resnet':  resnetModel
}
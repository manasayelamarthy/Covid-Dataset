import torch
from models import all_models
import cv2
import numpy as np
from utils import load_checkpoint
from config import train_Config

class Tester:
    def __init__(self, model:str, checkpoint_path:str, device):
        self.model = all_models[model](train_Config()).to(device)

        self.model = load_checkpoint(self.model, checkpoint_path)
        self.device = device

        self.label_maps = {
            0 : 'Covid',
            1 : 'Normal',
            2 : 'Viral Pneumonia' 
        }

    def predict(self, images: list[np.ndarray]):
        images = self.preprocess(images).to(self.device)
        outputs = self.model(images)
        preds = outputs.softmax(dim = 1).argmax(dim = 1).tolist()
        

        preds = [self.label_maps[idx] for idx in preds]

        return preds

    def preprocess(images: np.ndarray, image_size = (224,224)):
        """
        Apply Preprocessing to the given batch in the dataset 
        """
        images = cv2.resize(images, image_size)
        images = images/255.0

        return (torch.tensor(images).to(torch.float32).unsqueeze(0))

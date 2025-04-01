import torch
from models import all_models
import cv2
import numpy as np
from utils import load_checkpoint
from config import train_Config

class Tester:
    def __init__(self, model:str, checkpoint_path:str, device):
        self.model = all_models[model](train_Config()).model.to(device)

        self.model = load_checkpoint(self.model, checkpoint_path)
        self.device = device

        self.label_maps = {
            0 : 'Covid',
            1 : 'Normal',
            2 : 'Viral Pneumonia' 
        }

    def predict(self, images: list[np.ndarray]):
        images = self.preprocess(images).to(self.device)
        with torch.no_grad():
            outputs = self.model(images)
            probs = outputs.softmax(dim = 1)
        
            preds = probs.argmax(dim = 1).tolist()

        confidences = []
        for i, sample in enumerate(probs):
            confidences.append(round(sample[preds[i]].item()*100, 2))

        labels = [self.label_maps[idx] for idx in preds]

        return labels, confidences

    def preprocess(self, images: np.ndarray, image_size = (224,224)):
        """
        Apply Preprocessing to the given batch in the dataset 
        """
        images = [cv2.resize(img, image_size) for img in images]
        images = np.array(images)/255.0

        return (torch.tensor(images).to(torch.float32).unsqueeze(1))

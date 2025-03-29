import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging

import torch
from torch.utils.data import Dataset

#initialize logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok = True)

logger = logging.getLogger("data Ingestion")
logger.setLevel(logging.DEBUG)

console_logger = logging.StreamHandler()
console_logger.setLevel(logging.DEBUG)

file_path = os.path.join(log_dir, 'data Ingestion.log')
file_logger = logging.FileHandler(file_path)
file_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
console_logger.setFormatter(formatter)
file_logger.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_logger)



class covidData_ingestion():
    def __init__(self, root_dir):
        self.path = root_dir
        self.imagePaths = {}


    def load_paths(self):
        """
        Load image paths from root dir having 3 different classes.
        """
         
        num_classes = sorted(os.listdir(self.path))  
        class_to_index = {label: idx for idx, label in enumerate(num_classes)}  
        try :
            for label in num_classes:  
                class_dir = os.path.join(self.path, label)
                
                if os.path.isdir(class_dir):
                    for image_file in os.listdir(class_dir): 
                        image_path = os.path.join(class_dir, image_file)
                        # One-hot encoded label
                        one_hot_label = [0] * len(num_classes)
                        one_hot_label[class_to_index[label]] = 1
                        
                        self.imagePaths[image_path] = one_hot_label
                
        
                        
            logger.debug("loaded imagePaths with 3 classes")
            return self.imagePaths
        except Exception as e:
            logger.error('unable to load imagePaths %s',e)
        
        


    def plot_images(self):
            """
            Plot images from the dataset
            """
            image_paths = self.load_paths()

            covid_imagepath = [ path for path in image_paths if image_paths[path] == 'Covid'][0]
            normal_imagepath = [ path for path in image_paths if image_paths[path] == 'Normal'][0]
            viralp_imagepath = [ path for path in image_paths if image_paths[path] == 'Viral Pneumonia'][0]

            covidImage  = cv2.imread(covid_imagepath, cv2.IMREAD_GRAYSCALE)
            NormalImage = cv2.imread(normal_imagepath, cv2.IMREAD_GRAYSCALE)
            viralPImage = cv2.imread(viralp_imagepath, cv2.IMREAD_GRAYSCALE)

            _,axes =  plt.subplots(1, 3, figsize = (10, 5) )
            axes[0].imshow(covidImage, cmap = 'gray')
            axes[0].set_title("covidImage")
            axes[0].axis("off")

            axes[1].imshow(NormalImage, cmap = 'gray')
            axes[1].set_title("NormalImage")
            axes[1].axis("off")

            axes[2].imshow(viralPImage, cmap = 'gray')
            axes[2].set_title("viralPImage")
            axes[2].axis("off")

            plt.show()


    def preprocess(self, images: np.array, labels: list[str], image_size = (224,224)):
        """
        Apply Preprocessing to the given batch in the dataset 
        """
        try :
            images = cv2.resize(images, image_size)
            images = images/255.0

            logger.debug('preprocessed images')
            return images, labels
        except Exception as e:
            logger.error('preprocessing failed %s', e)

    

class CovidDataset(Dataset):
    def __init__(self, datapath :str, image_size, **args):
        self.data_ingester = covidData_ingestion(root_dir = datapath)
        self.image_paths = self.data_ingester.load_paths()
        self.image_size = image_size
        self.label_maps = {
            'Covid' : 0,
            'Normal': 1,
            'Viral Pneumonia' : 2 
        }


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = list(self.image_paths.keys())[idx]
        label = list(self.image_paths.values())[idx]
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
       
        image, label = self.data_ingester.preprocess(image, label, self.image_size)
        
        return torch.tensor(image).to(torch.float32), torch.tensor(label).to(torch.float32)



if __name__ =="__main__":
    pass
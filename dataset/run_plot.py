import sys
import os
from covid_dataset import covidDataset


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

datapath = "D:/Projects/Covid Image Dataset/data/covid/Covid19-dataset/train"
dataset = covidDataset(root_dir = datapath)

dataset.plot_images()
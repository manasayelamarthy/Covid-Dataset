import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import get_dataloaders
from config import train_Config
from models import all_models
from validation import validate

from utils import trainLogging,all_metrics,save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type = int, help = 'version of the config')
    parser.add_argument('--mode', choices = ['train', 'validation', 'test'], help = 'choose between train, validation, test')

    parser.add_argument('--data-dir', type = str, help = 'path to the dataset dir')
    parser.add_argument('--image-size', type = int, nargs = 2, help = 'dimensions of image') 
    parser.add_argument('--batch-size', type = int, help = 'batch size')


    parser.add_argument('--model', choices = ['cnn_model', 'efficient_net', 'resnet'], help = 'choose between models')
    parser.add_argument('--num_classes', type = int, help = 'num_classes')
    parser.add_argument('--optimizer', type = str, help = 'optimizer')
    parser.add_argument('--metrics', choices = ['accuracy', 'precision', 'recall'], help = 'choose between metrics')
    parser.add_argument('--learning_rate', type = int, help = 'learning_rate')
    parser.add_argument('--loss', type = int, help = 'loss')
    parser.add_argument('--epochs', type = int, help = 'number of epochs')
    parser.add_argument('--checkpoint-dir', type = str, help = 'path to the checkpoint dir')
    parser.add_argument('--log-dir', type = str, help = 'path to the log dir')

    return parser.parse_args()

args = arg_parse().__dict__
config = train_Config(**args)

print(config.model)
train_dataloader, val_dataloader = get_dataloaders(config)

trainer = all_models[config.model](config)

model = trainer.model.to(device)
train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Trainable parameters: " , train_parameters)

non_train_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print("Model Non-Trainable parameters: " , non_train_parameters)

optimizer = trainer.optimizer

criterion = nn.CrossEntropyLoss()
num_epochs = config.epochs

train_logs = {
    'loss' : 0,
    **{metric : 0 for metric in all_metrics}
}
best_accuracy = 0
metrics = list(metric for metric in all_metrics)
train_logger = trainLogging(metrics, config )

start_time = time.time()

for epoch in range(num_epochs):
    
    model.train()

    train_iterator = tqdm(train_dataloader, total = len(train_dataloader), desc = f'Epoch-{epoch+1}:')

    all_preds = []
    all_labels = []
    for inputs, labels in train_iterator:
        inputs = inputs.to(device)
        all_labels += labels.tolist()
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()
        train_logs['loss'] +=loss.item()

        preds = torch.argmax(torch.softmax(outputs, dim = 1), dim = 1)
        all_preds += preds.cpu().tolist()

    for metric in all_metrics:
        train_logs[metric]  = all_metrics[metric](all_preds, all_labels)
    train_logs['loss'] /= len(train_dataloader)


    val_logs = validate(model, val_dataloader, criterion, device)
    print("Train : ", train_logs)
    print("Validation : " , val_logs)

    train_logger.add_logs(epoch + 1, train_logs, val_logs)

    if val_logs['accuracy'] > best_accuracy:
        filename = config.checkpoint_dir + f'{config.model}.pth'
        checkpoint = save_checkpoint(model, filename)
        best_accuracy = val_logs['accuracy']

filename = config.log_dir + config.model + '.csv'
train_logger.save_logs(filename)

total_training_time = time.time() - start_time

print(f"training completed in {total_training_time:.2f}s")
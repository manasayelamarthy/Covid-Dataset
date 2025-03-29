import torch
from utils import all_metrics

def validate(model, val_dataloader, criterion):
    device = model.device
    logs = {
        'loss' : 0,
        **{metric : 0 for metric in all_metrics}
    }

    model.eval()

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            logs['loss'] +=loss.item()

            preds = torch.softmax(outputs, dim = 1)
            for metric in all_metrics.values():
                logs[metric] += metric(preds, labels)

    return logs  

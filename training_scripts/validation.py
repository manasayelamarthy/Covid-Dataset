import torch
from utils import all_metrics

def validate(model, val_dataloader, criterion, device):
    logs = {
        'loss' : 0,
        **{metric : 0 for metric in all_metrics}
    }

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            all_labels += labels.tolist()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())

            logs['loss'] +=loss.item()

            preds = torch.argmax(torch.softmax(outputs, dim = 1), dim = 1)
            all_preds += preds.cpu().tolist()
        for metric in all_metrics:
            logs[metric] = all_metrics[metric](all_preds, all_labels)

        logs['loss'] /= len(val_dataloader)
    return logs  

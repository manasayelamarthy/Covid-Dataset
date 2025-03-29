from .config import train_config
import pandas as pd

class trainLogging:
    def __init__(self, metrics:list[str], config = train_config()):
        self.config = config
        columns = ['epoch', 'train_loss', 'val_loss']

        train_metric_log = ['train_'+ metric for metric in metrics]
        val_metric_log   = ['val_'+ metric for metric in metrics]

        columns = columns + train_metric_log + val_metric_log

        self.logs : pd.Dataframe = pd.DataFrame(columns = columns)

def add_logs(self, epoch, train_logs, val_logs):
    epoch_row = {'epochs': epoch}

    epoch_row.update({f'train_{metric}':value for metric,value in train_logs.items()})
    epoch_row.update({f'val_{metric}':value for metric,value in val_logs.item()})

    self.logs = pd.concat([self.logs + pd.DataFrame(epoch_row)])

def save_logs(self, filename: str = None):
    self.logs = self.logs.sort_values(by = 'epoch')
    self.logs.to_csv(filename, index = False)
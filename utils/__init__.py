from .metrics import *
from .loggers import trainLogging
from helpers import *

all_metrics = {
    'accuracy'  : Accuracy,
    'precision' :Precision,
    'recall'    : Recall
}


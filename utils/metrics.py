from sklearn.metrics import accuracy_score, precision_score, recall_score

class Accuracy:
    def __init__(self):
        pass

    def __call__(self, preds, targets):
        return accuracy_score(targets, preds)

class Precision:
    def __init__(self, average="macro"):
        self.average = average

    def __call__(self, preds, targets):
        return precision_score(targets, preds, average=self.average, zero_division=1)

class Recall:
    def __init__(self, average="macro"):
        self.average = average

    def __call__(self, preds, targets):
        return recall_score(targets, preds, average=self.average, zero_division=1)

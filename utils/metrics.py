from sklearn.metrics import accuracy_score, precision_score, recall_score

class Accuracy:
    def __init__(self):
        pass

    def metric(self, preds, targets):
        preds = preds.argmax(dim=1)  
        return accuracy_score(targets, preds)

class Precision:
    def __init__(self, average="macro"):
        self.average = average

    def metric(self, preds, targets):
        preds = preds.argmax(dim=1)
        return precision_score(targets, preds, average=self.average)

class Recall:
    def __init__(self, average="macro"):
        self.average = average

    def metric(self, preds, targets):
        preds = preds.argmax(dim=1)
        return recall_score(targets, preds, average=self.average)

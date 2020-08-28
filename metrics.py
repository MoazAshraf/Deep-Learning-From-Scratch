import numpy as np


class Metric(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass


class BinaryAccuracy(Metric):
    def __init__(self):
        self.name = "accuracy"

    def call(self, y_true, y_pred):
        y_pred = (y_pred >= 0.5).astype(np.int)
        return np.mean((y_true == y_pred).astype(np.int))


class CategoricalAccuracy(Metric):
    def __init__(self):
        self.name = "accuracy"
    
    def call(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        return np.mean((y_true == y_pred).astype(np.int))


METRICS = {
    "binary_accuracy": BinaryAccuracy,
    "categorical_accuracy": CategoricalAccuracy,
    # "acc": BinaryAccuracy
}

print(METRICS)
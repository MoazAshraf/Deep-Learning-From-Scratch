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


METRICS = {
    "accuracy": BinaryAccuracy,
    "acc": BinaryAccuracy
}
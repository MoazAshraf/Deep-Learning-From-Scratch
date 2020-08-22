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
        m = y_true.shape[0]
        y_pred = (y_pred.astype(np.float64) >= 0.5).astype(np.int)
        return (np.sum((y_true == y_pred).astype(np.int)) / m).astype(np.float64)


METRICS = {
    "accuracy": BinaryAccuracy,
    "acc": BinaryAccuracy
}
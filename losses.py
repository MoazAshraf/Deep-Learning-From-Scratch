import numpy as np


class Loss(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass


class MSE(Loss):
    def call(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred), axis=-1)
    
    def derivative(self, y_true, y_pred):
        return 2 * (y_true - y_pred)


class BinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        return np.mean(-(y_true.T @ np.log(y_pred) + (1 - y_true).T @ np.log(1 - y_pred)), axis=-1)
    
    def derivative(self, y_true, y_pred):
        return (1 - y_true) / (1 - y_pred) - y_true / y_pred


LOSSES = {
    'mse': MSE,
    'mean_squared_error': MSE,
    'binary_crossentropy': BinaryCrossentropy
}
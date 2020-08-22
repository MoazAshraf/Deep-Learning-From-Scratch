import numpy as np


class Loss(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass


class MSE(Loss):
    def call(self, y_true, y_pred):
        m = y_true.shape[0]
        errors = y_true - y_pred
        return (errors.T @ errors / m).astype(np.float64)
    
    def derivative(self, y_true, y_pred):
        return 2 * (y_true - y_pred).astype(np.float64)


class BinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -(y_true.T @ np.log(y_pred) + (1 - y_true).T @ np.log(1 - y_pred)) / m
        return np.float64(np.squeeze(loss))
    
    def derivative(self, y_true, y_pred):
        deriv = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
        return deriv.astype(np.float64)


LOSSES = {
    'mse': MSE,
    'mean_squared_error': MSE,
    'binary_crossentropy': BinaryCrossentropy
}
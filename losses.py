import numpy as np


class Loss(object):
    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)

    def call(self, y_true, y_pred):
        pass


class MSE(Loss):
    def call(self, y_true, y_pred):
        m = y_true.shape[0]
        errors = y_true - y_pred
        return errors.T @ errors / m
    
    def derivative(self, y_true, y_pred):
        return 2 * (y_true - y_pred)


class BinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -(y_true.T @ np.log(y_pred) + (1 - y_true).T @ np.log(1 - y_pred)) / m
        return np.float64(np.squeeze(loss))
    
    def derivative(self, y_true, y_pred):
        deriv = -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
        return deriv


LOSSES = {
    'mse': MSE,
    'mean_squared_error': MSE,
    'binary_crossentropy': BinaryCrossentropy
}
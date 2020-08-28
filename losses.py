import numpy as np


class Loss(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass


class MSE(Loss):
    def call(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def derivative(self, y_true, y_pred):
        """
        Returns the derivative with respect to y_pred
        """

        m = y_true.shape[0]
        return -2 * (y_true - y_pred) / m


class BinaryCrossentropy(Loss):
    def call(self, y_true, y_pred):
        m = y_true.shape[0]
        return np.squeeze(-(y_true.T @ np.log(y_pred) + (1 - y_true).T @ np.log(1 - y_pred)) / m)
    
    def derivative(self, y_true, y_pred):
        """
        Returns the derivative with respect to y_pred
        """

        m = y_true.shape[0]
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / m


class CategoricalCrossentropy(Loss):
    def call(self, y_true, y_pred, epsilon=1e-12):
        m = y_true.shape[0]

        y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / m
    
    def derivative(self, y_true, y_pred, epsilon=1e-12):
        """
        Returns the derivative with respect to y_pred
        """

        m = y_true.shape[0]

        y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -(y_true / y_pred) / m


LOSSES = {
    'mse': MSE,
    'mean_squared_error': MSE,
    'binary_crossentropy': BinaryCrossentropy,
    'categorical_crossentropy': CategoricalCrossentropy
}
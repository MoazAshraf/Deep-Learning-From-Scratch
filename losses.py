import numpy as np
from sklearn.metrics import mean_squared_error


def binary_crossentropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -(y_true.T @ np.log(y_pred) + (1 - y_true).T @ np.log(1 - y_pred)) / m
    return np.float64(np.squeeze(loss))

def binary_crossentropy_loss_deriv(y_true, y_pred):
    m = y_true.shape[0]
    deriv = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / m
    return deriv

def mean_squared_error_deriv(y_true, y_pred):
    m = y_true.shape[0]
    return 2 * (y_true - y_pred) / m

LOSSES = {
    'mse': mean_squared_error,
    'mean_squared_error': mean_squared_error,
    'binary_crossentropy': binary_crossentropy_loss
}

LOSS_DERIVS = {
    'mse': mean_squared_error_deriv,
    'mean_squared_error': mean_squared_error_deriv,
    'binary_crossentropy': binary_crossentropy_loss_deriv
}
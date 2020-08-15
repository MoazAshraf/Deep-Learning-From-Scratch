import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    a = sigmoid(x)
    return a * (1 - a)

def tanh_deriv(x):
    return 1 - np.power(np.tanh(x), 2)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(np.float64)

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh': np.tanh,
    'relu': relu,
}

ACTIVATION_DERIVS = {
    'sigmoid': sigmoid_deriv,
    'tanh': tanh_deriv,
    'relu': relu_deriv
}
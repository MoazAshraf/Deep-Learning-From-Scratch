import numpy as np


class Activation(object):
    def __call__(self, x):
        return (self.call(x)).astype(np.float64)

    def call(self, x):
        pass


class Sigmoid(Activation):
    def call(self, x):
        return (1 / (1 + np.exp(-x))).astype(np.float64)
    
    def derivative(self, x):
        a = self.call(x)
        return (a * (1 - a)).astype(np.float64)


class Tanh(Activation):
    def call(self, x):
        return (np.tanh(x)).astype(np.float64)
    
    def derivative(self, x):
        return (1 - np.power(np.tanh(x), 2)).astype(np.float64)


class ReLU(Activation):
    def call(self, x):
        return np.maximum(0, x).astype(np.float64)

    def derivative(self, x):
        return (x > 0).astype(np.float64)

ACTIVATIONS = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
}
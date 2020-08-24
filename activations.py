import numpy as np


class Activation(object):
    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        pass


class Sigmoid(Activation):
    def call(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        a = self.call(x)
        return a * (1 - a)


class Tanh(Activation):
    def call(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.square(np.tanh(x))


class ReLU(Activation):
    def call(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(np.float64)

ACTIVATIONS = {
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
}
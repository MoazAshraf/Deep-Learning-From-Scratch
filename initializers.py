import numpy as np


def random_normal(shape, mean=0.0, std=0.05):
    X = np.random.randn(*shape) * std + mean
    return X

def he_normal(shape):
    input_shape = shape[0]
    X = np.random.randn(*shape) * np.sqrt(2 / input_shape)
    return X
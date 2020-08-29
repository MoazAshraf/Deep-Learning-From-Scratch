import numpy as np


def binary_accuracy(Y_true, Y_pred):
    """
    Expects Y_true to contain values of either 0 or 1 and Y_pred to be the probability
    of being 1
    """
    Y_pred = (Y_pred >= 0.5).astype(np.int)
    accuracy = np.mean((Y_true == Y_pred).astype(np.float32))
    return accuracy


def categorical_accuracy(Y_true, Y_pred):
    """
    Expects Y_true to be one-hot encoded and Y_pred to be normalized probabilities (e.g.
    the output of a Softmax layer)
    """
    Y_true = np.argmax(Y_true, axis=-1)
    Y_pred = np.argmax(Y_pred, axis=-1)
    accuracy = np.mean((Y_true == Y_pred).astype(np.float32))
    return accuracy
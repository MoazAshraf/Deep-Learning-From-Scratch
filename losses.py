import numpy as np


class Loss(object):
    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)
    
    def forward(self, Y_true, Y_pred, *args, **kwargs):
        pass
    
    def backward(self, Y_true, Y_pred, *args, **kwargs):
        """
        Computes the gradient of the loss with respect to Y_pred
        """
        pass


class MSE(Loss):
    def forward(self, Y_true, Y_pred):
        J = np.mean(np.square(Y_pred - Y_true))
        return J
    
    def backward(self, Y_true, Y_pred):
        size = Y_true.size
        dJ_dY_pred = 2 * (Y_pred - Y_true) / size
        return dJ_dY_pred


class BinaryCrossentropy(Loss):
    def forward(self, Y_true, Y_pred):
        J = -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))
        return J
    
    def backward(self, Y_true, Y_pred):
        size = Y_true.size
        dJ_dY_pred = ((1 - Y_true) / (1 - Y_pred) - Y_true / Y_pred) / size
        return dJ_dY_pred


class CategoricalCrossentropy(Loss):
    """
    Expects Y_true to be one-hot encoded and Y_pred to be normalized probabilities (e.g.
    the output of a Softmax layer)
    """
    def forward(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        J =  -np.sum(Y_true * np.log(Y_pred)) / m
        return J
    
    def backward(self, Y_true, Y_pred):
        m = Y_true.shape[0]
        dJ_dY_pred = -(Y_true / Y_pred) / m
        return dJ_dY_pred
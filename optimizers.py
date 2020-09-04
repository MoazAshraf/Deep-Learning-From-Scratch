import numpy as np


class Optimizer(object):
    pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    If momentum_beta is zero then vanilla SGD will be used:
    w = w - learning_rate * dJ_dw

    If momentum_beta is not zero, the optimizer will use momentum:
    v = momentum_beta * v + (1 - momentum_beta) * dJ_dw
    w = w - learning_rate * v
    """

    def __init__(self, learning_rate=0.01, momentum_beta=0.0):
        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
    
    def optimize(self, w, dJ_dw, optimizer_params=None):
        if self.momentum_beta == 0:
            # SGD with no momentum
            w = w - self.learning_rate * dJ_dw
        else:
            # SGD with momentum
            if optimizer_params is None:
                optimizer_params = {'velocity': np.zeros(w.shape)}
            velocity = optimizer_params['velocity']

            velocity = self.momentum_beta * velocity + (1 - self.momentum_beta) * dJ_dw
            w = w - self.learning_rate * velocity

            optimizer_params['velocity'] = velocity
        return w, optimizer_params


class RMSprop(Optimizer):
    """
    RMSprop

    The update rule is:
    S = beta * S + (1 - beta) * dJ_dw ** 2
    w = w - learning_rate * dJ_dw / sqrt(S)
    """

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
    
    def optimize(self, w, dJ_dw, optimizer_params=None):        
        if optimizer_params is None:
            optimizer_params = {'S': np.zeros(w.shape)}
        S = optimizer_params['S']
        
        S = self.beta * S + (1 - self.beta) * np.square(dJ_dw)
        w = w - self.learning_rate * dJ_dw / (np.sqrt(S) + self.epsilon)

        optimizer_params['S'] = S
        return w, optimizer_params



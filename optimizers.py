import numpy as np


class Optimizer(object):
    pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    If momentum_beta is not zero, the optimizer will use momentum. Otherwise, it will use
    vanilla gradient descent.
    """

    def __init__(self, learning_rate=0.01, momentum_beta=0.0):
        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
    
    def optimize(self, θ, dJ_dθ, optimizer_params=None):
        if self.momentum_beta == 0:
            # SGD with no momentum
            θ = θ - self.learning_rate * dJ_dθ
        else:
            # SGD with momentum
            if optimizer_params is None:
                optimizer_params = {'velocity': np.zeros(θ.shape)}
            velocity = optimizer_params['velocity']
            velocity = self.momentum_beta * velocity + (1 - self.momentum_beta) * dJ_dθ
            θ = θ - self.learning_rate * velocity
            optimizer_params['velocity'] = velocity
        return θ, optimizer_params


class RMSprop(Optimizer):
    """
    RMSprop
    """

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
    
    def optimize(self, θ, dJ_dθ, optimizer_params=None):
        if optimizer_params is None:
            optimizer_params = {'S': np.zeros(θ.shape)}
        S = optimizer_params['S']
        S = self.beta * S + (1 - self.beta) * np.square(dJ_dθ)
        θ = θ - self.learning_rate * dJ_dθ / (np.sqrt(S) + self.epsilon)
        optimizer_params['S'] = S
        return θ, optimizer_params
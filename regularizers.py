import numpy as np


class Regularizer(object):
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        pass


class L1(Regularizer):
    """
    L1 regularization, also known as Lasso regularization

    Computed as l1 * sum(abs(x))
    """

    def __init__(self, l1=0.01):
        self.l1 = l1

    def call(self, x):
        return self.l1 * np.sum(np.abs(x))
    
    def derivative(self, x):
        dx = (x > 0).astype(np.int) * 2 - 1
        dx[x == 0] = 0
        return self.l1 * dx


class L2(Regularizer):
    """
    L2 regularization, also known as Ridge regularization

    Computed as l2 * sum(square(x))
    """

    def __init__(self, l2=0.01):
        self.l2 = l2
    
    def call(self, x):
        return self.l2 * np.sum(np.square(x))
    
    def derivative(self, x):
        return self.l2 * 2 * x


class L1L2(Regularizer):
    """
    Applies both L1 and L2 regularization

    Computed as l1 * sum(abs(x)) + l2 * sum(square(x))
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

        self.l1_regularizer = L1(l1)
        self.l2_regularizer = L2(l2)
    
    def call(self, x):
        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2
        return self.l1_regularizer(x) + self.l2_regularizer(x)
    
    def derivative(self, x):
        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2
        return self.l1_regularizer.derivative(x) + self.l2_regularizer.derivative(x)
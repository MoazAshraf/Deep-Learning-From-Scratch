import numpy as np


class Regularizer(object):
    def __call__(self, w):
        return self.forward(w)

    def forward(self, w):
        pass

    def backward(self, w):
        """
        Computes the gradient of the loss with respect to w
        """
        pass


class L1(Regularizer):
    """
    L1 regularization, also known as Lasso regularization

    Computed as l1 * sum(abs(w))
    """

    def __init__(self, l1=0.01):
        self.l1 = l1

    def forward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 to depend on m
        """

        J = self.l1 * np.sum(np.abs(w))
        return J
    
    def backward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 to depend on m
        """

        # dw = (w > 0).astype(np.int) * 2 - 1
        # dw[w == 0] = 0
        dJ_dw = self.l1 * np.sign(w)
        return dJ_dw


class L2(Regularizer):
    """
    L2 regularization, also known as Ridge regularization

    Computed as l2 * sum(square(w))
    """

    def __init__(self, l2=0.01):
        self.l2 = l2
    
    def forward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l2 to depend on m
        """

        J = self.l2 * np.sum(np.square(w))
        return J
    
    def backward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l2 to depend on m
        """

        dJ_dw = self.l2 * 2 * w
        return dJ_dw


class L1L2(Regularizer):
    """
    Applies both L1 and L2 regularization

    Computed as l1 * sum(abs(w)) + l2 * sum(square(w))
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

        self.l1_regularizer = L1(l1)
        self.l2_regularizer = L2(l2)
    
    def forward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 and l2 to depend on m
        """

        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2

        J = self.l1_regularizer(w) + self.l2_regularizer(w)
        return J
    
    def backward(self, w):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 and l2 to depend on m
        """

        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2

        dJ_dw = self.l1_regularizer.backward(w) + self.l2_regularizer.backward(w)
        return dJ_dw
import numpy as np


class Regularizer(object):
    def __call__(self, θ):
        return self.forward(θ)

    def forward(self, θ):
        pass

    def backward(self, θ):
        """
        Computes the gradient of the loss with respect to θ
        """
        pass


class L1(Regularizer):
    """
    L1 regularization, also known as Lasso regularization

    Computed as l1 * sum(abs(θ))
    """

    def __init__(self, l1=0.01):
        self.l1 = l1

    def forward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 to depend on m
        """

        J = self.l1 * np.sum(np.abs(θ))
        return J
    
    def backward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 to depend on m
        """

        # dθ = (θ > 0).astype(np.int) * 2 - 1
        # dθ[θ == 0] = 0
        dJ_dθ = self.l1 * np.sign(θ)
        return dJ_dθ


class L2(Regularizer):
    """
    L2 regularization, also known as Ridge regularization

    Computed as l2 * sum(square(θ))
    """

    def __init__(self, l2=0.01):
        self.l2 = l2
    
    def forward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l2 to depend on m
        """

        J = self.l2 * np.sum(np.square(θ))
        return J
    
    def backward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l2 to depend on m
        """

        dJ_dθ = self.l2 * 2 * θ
        return dJ_dθ


class L1L2(Regularizer):
    """
    Applies both L1 and L2 regularization

    Computed as l1 * sum(abs(θ)) + l2 * sum(square(θ))
    """

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2

        self.l1_regularizer = L1(l1)
        self.l2_regularizer = L2(l2)
    
    def forward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 and l2 to depend on m
        """

        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2

        J = self.l1_regularizer(θ) + self.l2_regularizer(θ)
        return J
    
    def backward(self, θ):
        """
        Note: you have to divide the result by m, the batch size, if you don't want l1 and l2 to depend on m
        """

        self.l1_regularizer.l1, self.l2_regularizer.l2 = self.l1, self.l2

        dJ_dθ = self.l1_regularizer.backward(θ) + self.l2_regularizer.backward(θ)
        return dJ_dθ
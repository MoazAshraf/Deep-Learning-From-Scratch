class Optimizer(object):
    pass


class SGD(Optimizer):
    """
    Vanilla Gradient Descent
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def optimize(self, θ, dJ_dθ):
        θ = θ - self.learning_rate * dJ_dθ
        return θ
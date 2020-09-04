import numpy as np


def collapse_to_zero(x, tol=1e-16, inplace=True):
    """
    If the absolute value of x is less than tol, set it to zero. This function is
    applied element-wise. It is used to prevent multiplication underflow.
    """

    if not inplace:
        x = x.copy()
    x[np.abs(x) < tol] = 0.0
    return x


class Optimizer(object):
    pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    If momentum_beta is zero then vanilla SGD will be used:
    w = w - learning_rate * dJ_dw

    If momentum_beta is not zero, the optimizer will use momentum:
    m = momentum_beta * m + (1 - momentum_beta) * dJ_dw
    w = w - learning_rate * m
    """

    def __init__(self, learning_rate, momentum_beta=0.0):
        self.learning_rate = learning_rate
        self.momentum_beta = momentum_beta
    
    def optimize(self, w, dJ_dw, vars=None):
        if self.momentum_beta == 0:
            # SGD with no momentum
            w = w - self.learning_rate * dJ_dw
        else:
            # SGD with momentum
            if vars is None:
                vars = {'m': np.zeros(w.shape)}
            m = vars['m']

            m = self.momentum_beta * m + (1 - self.momentum_beta) * dJ_dw
            w = w - self.learning_rate * m

            vars['m'] = m
        return w, vars


class RMSprop(Optimizer):
    """
    RMSprop

    The update rule is:
    v = beta * v + (1 - beta) * dJ_dw ** 2
    w = w - learning_rate * dJ_dw / (sqrt(v) + epsilon)
    """

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
    
    def optimize(self, w, dJ_dw, vars=None):        
        if vars is None:
            vars = {'v': np.zeros(w.shape)}
        v = vars['v']

        v = self.beta * v + (1 - self.beta) * np.square(dJ_dw)
        w = w - self.learning_rate * dJ_dw / (np.sqrt(v) + self.epsilon)

        vars['v'] = v
        return w, vars


class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation

    The update rule is:
    m = beta_1 * m + (1 - beta_1) * dJ_dw
    v = beta_2 * v + (1 - beta_2) * dJ_dw ** 2
    m_corrected = m / (1 - beta_1 ** t)
    v_corrected = v / (1 - beta_2 ** t)
    w = w - learning_rate * m_corrected / (sqrt(v_corrected) + epsilon)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
    
    def optimize(self, w, dJ_dw, vars=None):
        if vars is None:
            vars = {
                't': 1,
                'm': np.zeros(w.shape),
                'v': np.zeros(w.shape)
            }
        t = vars['t']
        m = vars['m']
        v = vars['v']

        m = (self.beta_1 * m + (1 - self.beta_1) * dJ_dw)
        v = (self.beta_2 * v + (1 - self.beta_2) * np.square(dJ_dw))
        m_corrected = m / (1 - self.beta_1 ** t)
        v_corrected = v / (1 - self.beta_2 ** t)
        w = w - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        vars['t'] += 1
        vars['m'] = m
        vars['v'] = v
        return w, vars
import numpy as np
from initializers import he_normal


class Layer(object):
    """
    An abstract class that represents a neural network layer.
    """
    
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.cache = {'X': None}
        self.optimizer_params = {}
    
    def build(self):
        """
        Initializes the layer's parameters that depend on the input shape
        """

        pass
    
    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)
    
    @staticmethod
    def forward(f):
        def forward(self, X, training=False, *args, **kwargs):
            """
            Computes the layer's output (forward pass)
            
            If training is True, the input and output of each layer is stored so that it
            can be used later during backprop. Also, some layers such as Dropout behave
            differently if training is set to True.
            """

            Z = f(self, X, training=training, *args, **kwargs)

            if training:
                self.cache['X'] = X
                self.cache['Z'] = Z
            
            return Z
        return forward
    
    @staticmethod
    def backward(f):
        def backward(self, dJ_dZ, training=False, *args, **kwargs):
            """
            Computes the gradients (backward pass) of the loss function with respect to the
            layer's parameters (if any) and the layer's input
            
            dJ_dZ is the tensor of gradients of the loss function with respect to the layer's
            outputs
            
            When implementing this function for a layer with parameters, you must return dJ_dX
            (the gradients with respect to the input) first in a tuple, then the rest of the
            tuple should contain the gradients with respect to the layer's parameters in the
            same order of the arguments in update_parameters()
            """

            return f(self, dJ_dZ, training=training, *args, **kwargs)
        return backward

    def update_parameters(self, optimizer):
        """
        Updates the layer's parameters
        """
        
        pass


class Linear(Layer):
    """
    Just your regular fully connected layer
    """
    
    def __init__(self, units, kernel_regularizer=None, kernel_initializer=he_normal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
    
    def build(self, weights=None, biases=None, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (self.units,)

        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.kernel_initializer((self.input_shape[0], self.units))
            
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.zeros((1, self.units))
        
        self.optimizer_params['weights'] = None
        self.optimizer_params['biases'] = None
    
    @Layer.forward
    def forward(self, X, training=False):
        Z = X @ self.weights + self.biases
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        # compute the gradients with respect to the weights
        X = self.cache['X']
        dJ_dW = X.T @ dJ_dZ
        # add regularization gradients
        if self.kernel_regularizer is not None:
            m = dJ_dZ.shape[0]
            dJ_dW += self.kernel_regularizer.backward(self.weights) / m
        
        # compute the gradients with respect to the biases
        dJ_db = np.sum(dJ_dZ, axis=0, keepdims=True)
        
        # compute the gradients with respect to the input
        dJ_dX = dJ_dZ @ self.weights.T
        
        return dJ_dX, dJ_dW, dJ_db
    
    def update_parameters(self, dJ_dW, dJ_db, optimizer):
        self.weights, self.optimizer_params['weights'] = optimizer.optimize(self.weights, dJ_dW,
                                                                            self.optimizer_params['weights'])
        self.biases, self.optimizer_params['biases'] = optimizer.optimize(self.biases, dJ_db,
                                                                          self.optimizer_params['biases'])


class ReLU(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape
        
    @Layer.forward
    def forward(self, X, training=False):
        Z = np.maximum(0, X)
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        X = self.cache['X']
        dJ_dX = dJ_dZ * np.heaviside(X, 0)
        return dJ_dX


class Tanh(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape
    
    @Layer.forward
    def forward(self, X, training=False):
        Z = np.tanh(X)
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        Z = self.cache['Z']
        dJ_dX = dJ_dZ * (1 - np.square(Z))
        return dJ_dX

class Sigmoid(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape

    @Layer.forward 
    def forward(self, X, training=False):
        Z = 1 / (1 + np.exp(-X))
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        Z = self.cache['Z']
        dJ_dX = dJ_dZ * Z * (1 - Z)
        return dJ_dX


class Softmax(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape
        
    @Layer.forward
    def forward(self, X, training=False):
        X = X - np.max(X, axis=-1, keepdims=True)
        exp_X = np.exp(X)
        Z = exp_X / exp_X.sum(axis=-1, keepdims=True)
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        # dJ_dX = dJ_dZ * Z - (dJ_dZ @ Z.T * np.eye(m)) @ Z  # -> needs a lot (m^2) of memory
        
        Z = self.cache['Z']
        dJ_dX = (dJ_dZ - np.sum(dJ_dZ * Z, axis=-1, keepdims=True)) * Z
        return dJ_dX


class Dropout(Layer):
    def __init__(self, drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.drop_rate = drop_rate

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape
    
    @Layer.forward
    def forward(self, X, training=False):
        if training:
            mask = (np.random.rand(*X.shape) >= self.drop_rate).astype(int)
            self.cache['mask'] = mask
            Z = X * mask / (1 - self.drop_rate)
        else:
            Z = X
        return Z
    
    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        if training:
            mask = self.cache['mask']
            dJ_dX = dJ_dZ * mask / (1 - self.drop_rate)
        else:
            dJ_dX = dJ_dZ
        return dJ_dX


class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-8, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

        self.epsilon = epsilon

    def build(self, gamma=None, beta=None, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = self.input_shape

        if beta is not None:
            self.beta = beta
        else:
            self.beta = np.zeros((1,) + self.input_shape)
        
        if gamma is not None:
            self.gamma = gamma
        else:
            self.gamma = np.ones((1,) + self.input_shape)
        
        self.optimizer_params['gamma'] = None
        self.optimizer_params['beta'] = None
    
    @Layer.forward
    def forward(self, X, training=False):
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_var = np.mean(np.square(X - X_mean), axis=0, keepdims=True)
        X_hat = (X - X_mean) / (np.sqrt(X_var) + self.epsilon)
        Z = self.gamma * X_hat + self.beta

        if training:
            self.cache['X_hat'] = X_hat
            self.cache['X_mean'] = X_mean
            self.cache['X_var'] = X_var
        return Z

    @Layer.backward
    def backward(self, dJ_dZ, training=False):
        X = self.cache['X']
        X_hat = self.cache['X_hat']
        X_mean = self.cache['X_mean']
        X_var = self.cache['X_var']
        
        m = X.shape[0]

        dJ_dgamma = np.sum(dJ_dZ * X_hat, axis=0, keepdims=True)
        dJ_dbeta = np.sum(dJ_dZ, axis=0, keepdims=True)
        
        dJ_dX_hat = dJ_dZ * self.gamma
        
        minus = X - X_mean
        denom = (np.sqrt(X_var) + self.epsilon)

        dminus_dX = 1 - X / m
        ddenom_dX = (2 / m) * (X - X_mean) * dminus_dX / denom
        dX_hat_dX = (denom * dminus_dX - minus * ddenom_dX) / np.square(denom)
        dJ_dX = dJ_dX_hat * dX_hat_dX

        return dJ_dX, dJ_dgamma, dJ_dbeta
    
    def update_parameters(self, dJ_dgamma, dJ_dbeta, optimizer):
        self.gamma, self.optimizer_params['gamma'] = optimizer.optimize(self.gamma, dJ_dgamma,
                                                                            self.optimizer_params['gamma'])
        self.beta, self.optimizer_params['beta'] = optimizer.optimize(self.beta, dJ_dbeta,
                                                                          self.optimizer_params['beta'])
import numpy as np
from activations import ACTIVATIONS, ACTIVATION_DERIVS


class Dense(object):
    """
    Just your regular densely-connected layer
    """

    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        if type(activation) is str:
            self.activation = ACTIVATIONS[activation]
            self.activation_deriv = ACTIVATION_DERIVS[activation]
        else:
            # TODO
            pass
        
        self.input_shape = input_shape
        self.output_shape = (self.units,)
        
    def build(self, input_shape):
        """
        Initializes the layer's weights and biases
        """

        self.input_shape = input_shape
        self.weights = np.random.randn(self.units, input_shape[0]) * 0.01
        self.biases = np.zeros((self.units, 1))
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
    
    def call(self, X, cache=False):
        """
        Generates the output of the layer for the given input samples

        If cache is True then the linear output (before being passed to the activation function) is stored;
        This is generally used during training.
        """

        z = X @ self.weights.T + self.biases.T
        if self.activation is not None:
            a = self.activation(z)
        if cache:
            self.cache = {'linear': z}
        return a
    
    def compute_gradients(self, dL_da, X):
        """
        Computes the gradients of the loss with respect to the layer's weights biases and inputs

        dL_da is the gradient of the loss with respect the layer's activations
        X is the input the layer (e.g. the activations of the previous layer)
        """

        m = X.shape[0]
        
        # calculates the gradients of the loss with respect to the layer's weights and biases
        dL_dz = dL_da * self.activation_deriv(self.cache['linear'])
        dL_dw = dL_dz.T @ X
        dL_db = np.sum(dL_dz, axis=0, keepdims=True).T / m

        # calculates the gradients of the loss with respect to the input
        dL_dX = dL_dz @ self.weights

        return dL_dw, dL_db, dL_dX
    
    def update_params(self, dL_dw, dL_db, learning_rate):
        """
        Updates the layer's parameters
        """

        self.weights = self.weights - learning_rate * dL_dw
        self.biases = self.biases - learning_rate * dL_db
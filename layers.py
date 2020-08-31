import numpy as np


class Layer(object):
    """
    An abstract class that represents a neural network layer.
    """
    
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.cache = {'X': None}
    
    def build(self):
        """
        Initializes the layer's parameters that depend on the input shape
        """

        pass
    
    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)
    
    def forward(self, X, cache=False, *args, **kwargs):
        """
        Computes the layer's output (forward pass)
        
        If cache is True, the input is stored so that it can be used later during
        backprop
        """

        Z = self._forward(X, *args, **kwargs)

        if cache:
            self.cache['X'] = X
            self.cache['Z'] = Z
        
        return Z

    def _forward(self, X):
        return X
    
    def backward(self, dJ_dZ, *args, **kwargs):
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

        return self._backward(dJ_dZ, *args, **kwargs)
    
    def _backward(self, dJ_dZ):
        return dJ_dZ

    def update_parameters(self, learning_rate):
        """
        Updates the layer's parameters
        """
        
        pass


class Linear(Layer):
    """
    Just your regular fully connected layer
    """
    
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.units = units
        self.output_shape = (units,)
    
    def build(self, weights=None, biases=None, *args, **kwargs):        
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.randn(self.input_shape[0], self.units) * np.sqrt(2 / self.input_shape[0])
            
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.zeros((1, self.units))
    
    def _forward(self, X):
        Z = X @ self.weights + self.biases
        return Z
    
    def _backward(self, dJ_dZ):
        # compute the gradients with respect to the weights and biases
        X = self.cache['X']
        dJ_dW = X.T @ dJ_dZ
        dJ_db = np.sum(dJ_dZ, axis=0, keepdims=True)
        
        # compute the gradients with respect to the input
        dJ_dX = dJ_dZ @ self.weights.T
        
        return dJ_dX, dJ_dW, dJ_db
    
    def update_parameters(self, dJ_dW, dJ_db, learning_rate):
        self.weights = self.weights - learning_rate * dJ_dW
        self.biases = self.biases - learning_rate * dJ_db


class ReLU(Layer):
    def build(self):
        self.output_shape = self.input_shape
        
    def _forward(self, X):
        Z = np.maximum(0, X)
        return Z
    
    def _backward(self, dJ_dZ):
        X = self.cache['X']
        dJ_dX = dJ_dZ * np.heaviside(X, 0)
        return dJ_dX


class Tanh(Layer):
    def build(self):
        self.output_shape = self.input_shape
    
    def _forward(self, X, *args, **kwargs):
        Z = np.tanh(X)
        return Z
    
    def _backward(self, dJ_dZ):
        Z = self.cache['Z']
        dJ_dX = dJ_dZ * (1 - np.square(Z))
        return dJ_dX

class Sigmoid(Layer):
    def build(self, *args, **kwargs):
        self.output_shape = self.input_shape
        
    def _forward(self, X, *args, **kwargs):
        Z = 1 / (1 + np.exp(-X))
        return Z
    
    def _backward(self, dJ_dZ):
        Z = self.cache['Z']
        dJ_dX = dJ_dZ * Z * (1 - Z)
        return dJ_dX


class Softmax(Layer):
    def build(self, *args, **kwargs):
        self.output_shape = self.input_shape
        
    def _forward(self, X, *args, **kwargs):
        X = X - np.max(X, axis=-1, keepdims=True)
        exp_X = np.exp(X)
        Z = exp_X / exp_X.sum(axis=-1, keepdims=True)
        return Z
    
    def _backward(self, dJ_dZ):
        # dJ_dX = dJ_dZ * Z - (dJ_dZ @ Z.T * np.eye(m)) @ Z  # -> needs a lot (m^2) of memory
        
        Z = self.cache['Z']
        dJ_dX = (dJ_dZ - np.sum(dJ_dZ * Z, axis=-1, keepdims=True)) * Z
        return dJ_dX
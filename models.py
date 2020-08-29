import numpy as np


class Model(object):
    """
    An abstract class that represents a machine learning model.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)
    
    def forward(self, X, cache=False, *args, **kwargs):
        """
        Computes the model's output (forward pass)
        
        If cache is True, the input to each layer is stored so that it can be used later during
        backprop
        """
        pass
    
    def configure(self, loss, learning_rate, metrics=None):
        """
        Configure the model for training or evaluation
        """
        self.loss = loss
        self.learning_rate = learning_rate
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = []
    
    def train_step(self, X, Y, learning_rate, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training batch (X, Y).
        """
        pass
    
    def train(self, X, Y, epochs=10, verbose=True, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training set for the given number of epochs (X, Y).
        """
        for epoch in range(epochs):
            self.train_step(X, Y, self.learning_rate, *args, **kwargs)
            
            if verbose:
                # compute the new predictions, the loss and specified metrics
                Y_pred = self.forward(X)
                J = self.loss(Y, Y_pred)
                metric_values = [metric(Y, Y_pred) for metric in self.metrics]

                epoch_info = f"Epoch {epoch+1:02}\t" + self.format_loss_and_metrics(J, metric_values)
                print(epoch_info)
    
    def evaluate(self, X, Y):
        """
        Evaluates the configured loss and metrics on the given data batch

        Returns the loss and the metrics
        """

        Y_pred = self.forward(X)
        J = self.loss(Y, Y_pred)
        metric_values = [metric(Y, Y_pred) for metric in self.metrics]

        eval_info = self.format_loss_and_metrics(J, metric_values)
        print(eval_info)

        return J, metric_values

    def format_loss_and_metrics(self, J, metric_values):
        metric_names = [metric.__name__ for metric in self.metrics]
        
        s = f"loss={J:.6f} \t"
        s += '\t'.join([f"{name}={value:.6f}" for name, value in zip(metric_names, metric_values)])
        return s

class Sequential(Model):
    def __init__(self, layers=None):
        self.layers = []
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer):
        """
        Add a layer to the model
        """
        if len(self.layers) == 0:
            if layer.input_shape is None:
                raise Exception("The input shape for the first layer of the model must be "
                                "specified")
        else:
            layer.input_shape = self.layers[-1].output_shape
        layer.build()
        self.layers.append(layer)
    
    def forward(self, X, cache=False):
        Y_pred = X
        for layer in self.layers:
            Y_pred = layer(Y_pred, cache=cache)
            
        return Y_pred
    
    def train_step(self, X, Y, learning_rate=0.01):
        # forward pass
        Y_pred = self.forward(X, cache=True)
        
        # compute the gradient of the loss with respect to the Y_pred
        dJ_dY = self.loss.backward(Y, Y_pred)
    
        # dJ_dZ is a variable that holds the gradient of the loss with respect to
        # the current layer's outputs
        dJ_dZ = dJ_dY
        for l in range(len(self.layers)-1, -1, -1):
            layer = self.layers[l]
            
            # backpropagate through this layer and update its parameters
            grads = layer.backward(dJ_dZ)
            if isinstance(grads, tuple):
                dJ_dZ = grads[0]
                layer.update_parameters(*grads[1:], learning_rate)
            else:
                dJ_dZ = grads
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

        return X
    
    def configure(self, loss, learning_rate, metrics=None):
        """
        Configure the model for training or evaluation

        metrics should be a dictionary with the name each of the metric as the key and the
        function that computes the metric as the value
        """

        self.loss = loss
        self.learning_rate = learning_rate
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = {}

    def _evaluate(self, Y_train, Y_train_pred, Y_val=None, Y_val_pred=None):
        metrics = {}
        metrics['loss'] = self.loss(Y_train, Y_train_pred)
        if Y_val is not None:
            metrics['val_loss'] = self.loss(Y_val, Y_val_pred)

        for name, metric in self.metrics.items():
            metrics[name] = metric(Y_train, Y_train_pred)
            if Y_val is not None:
                metrics[f"val_{name}"] = metric(Y_val, Y_val_pred)
        
        return metrics
        
    def evaluate(self, X, Y, X_val=None, Y_val=None):
        """
        Evaluates the configured loss and metrics on the given data batch

        Returns the loss and the metrics
        """

        # make predictions on the training set
        Y_pred = self.forward(X)

        # make predictions on the validation set
        if X_val is not None:
            Y_val_pred = self.forward(X_val)
        else:
            Y_val_pred = None
        
        # compute and print the metrics
        metrics = self._evaluate(Y, Y_pred, Y_val, Y_val_pred)
        eval_info = self.format_metrics(metrics)
        print(eval_info)
        return metrics

    def format_metrics(self, metrics):
        """
        metrics should be a dictionary with the name each of the metric as the key and the
        value of the metric as the dictionary value
        """

        s = '\t'.join([f"{name}={value:.6f}" for name, value in metrics.items()])
        return s

    def train_step(self, X, Y, learning_rate, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training batch (X, Y).
        """

        pass
    
    def train(self, X, Y, X_val=None, Y_val=None, epochs=10, verbose=True, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training set for the given number of epochs (X, Y).
        """
        
        for epoch in range(epochs):
            self.train_step(X, Y, self.learning_rate, *args, **kwargs)
            
            if verbose:
                # make predictions on the training set
                Y_pred = self.forward(X)

                # make predictions on the validation set
                if X_val is not None:
                    Y_val_pred = self.forward(X_val)
                else:
                    Y_val_pred = None
                
                # compute and print the metrics
                metrics = self._evaluate(Y, Y_pred, Y_val, Y_val_pred)
                epoch_info = f"Epoch {epoch+1:02}\t" + self.format_metrics(metrics)
                print(epoch_info)


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
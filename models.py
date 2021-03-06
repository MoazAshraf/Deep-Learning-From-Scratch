import numpy as np
from optimizers import SGD


class Model(object):
    """
    An abstract class that represents a machine learning model.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)
    
    def forward(self, X, training=False, *args, **kwargs):
        """
        Computes the model's output (forward pass)
        
        If training is True variables that need to be stored during training are kept in a cache
        dictionary for each layer. Also, some layers such as Dropout behave differently if
        training is set to True.
        """

        return X
    
    def configure(self, loss, optimizer=SGD(learning_rate=0.01), metrics=None):
        """
        Configure the model for training or evaluation

        metrics should be a dictionary with the name each of the metric as the key and the
        function that computes the metric as the value
        """

        self.loss = loss
        self.optimizer = optimizer
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

    def train_step(self, X, Y, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training batch (X, Y).
        """

        pass
    
    def train(self, X, Y, X_val=None, Y_val=None, epochs=10, batch_size=32, drop_remainder=True, shuffle=True, verbose=True, *args, **kwargs):
        """
        Performs backpropagation through the model and updates the model's parameters using the
        training set for the given number of epochs (X, Y).

        - X_val and Y_val can be set to evaluate the model on validation data after each epoch.
        - If drop_remainder is True, the last batch will be dropped if its size is less than
          batch_size.
        - If shuffle is True, the training set will be shuffled before each epoch.
        """

        history = []
        if batch_size > X.shape[0]:
            drop_remainder = False

        for epoch in range(epochs):
            # shuffle the training set
            if shuffle:
                indices = np.random.permutation(X.shape[0])
                X = X[indices]
                Y = Y[indices]

            # drop the last minibatch if its size is less than batch_sze
            if drop_remainder:
                train_size = (X.shape[0] // batch_size) * batch_size
            else:
                train_size = X.shape[0]

            # loop over each batch and train the model
            for batch_start in range(0, train_size, batch_size):
                batch_end = batch_start + batch_size
                X_batch = X[batch_start:batch_end]
                Y_batch = Y[batch_start:batch_end]
                self.train_step(X_batch, Y_batch, *args, **kwargs)
            
            # make predictions on the training set
            Y_pred = self.forward(X)

            # make predictions on the validation set
            if X_val is not None:
                Y_val_pred = self.forward(X_val)
            else:
                Y_val_pred = None
            
            # compute the metrics and store them in history
            metrics = self._evaluate(Y, Y_pred, Y_val, Y_val_pred)
            history.append(metrics)

            if verbose:
                epoch_info = f"Epoch {epoch+1:02}\t" + self.format_metrics(metrics)
                print(epoch_info)
        
        return history


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
    
    def forward(self, X, training=False):
        Y_pred = X
        for layer in self.layers:
            Y_pred = layer(Y_pred, training=training)
            
        return Y_pred
    
    def train_step(self, X, Y):
        # forward pass
        Y_pred = self.forward(X, training=True)
        
        # compute the gradient of the loss with respect to the Y_pred
        dJ_dY = self.loss.backward(Y, Y_pred)
    
        # dJ_dZ is a variable that holds the gradient of the loss with respect to
        # the current layer's outputs
        dJ_dZ = dJ_dY
        for l in range(len(self.layers)-1, -1, -1):
            layer = self.layers[l]
            
            # backpropagate through this layer and update its parameters
            grads = layer.backward(dJ_dZ, training=True)
            if isinstance(grads, tuple):
                dJ_dZ = grads[0]
                layer.update_parameters(*grads[1:], self.optimizer)
            else:
                dJ_dZ = grads
from losses import LOSSES
from metrics import METRICS


class Sequential(object):
    """
    A sequential model is made up of a stack of layers
    """

    def __init__(self, layers=None):
        self.compiled = False
        
        # initialize the layers of the model
        if layers is not None:
            self.layers = layers
            if len(self.layers) > 0:
                if self.layers[0].input_shape is None:
                    raise Exception("You must specify the input shape of the first layer")
                
                # initialize the input shapes, weights and biases of each layer
                self.input_shape = self.layers[0].input_shape
                self.layers[0].build(self.input_shape)
                for i in range(1, len(self.layers)):
                    self.layers[i].input_shape = self.layers[i-1].output_shape
                    self.layers[i].build(self.layers[i].input_shape)
        else:
            self.layers = []
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
    
    def call(self, X):
        """
        Generates output for the given input
        """

        for layer in self.layers:
            X = layer(X)
        return X
    
    def compile(self, loss, learning_rate=0.01, metrics=None):
        """
        Configures the model for training, evaluation and/or prediction
        """

        if isinstance(loss, str):
            self.loss = LOSSES[loss]()
        else:
            self.loss = loss
        
        self.metrics = []
        if metrics is not None:
            for metric in metrics:
                if isinstance(metric, str):
                    self.metrics.append(METRICS[metric]())
                else:
                    self.metrics.append(metric)

        self.learning_rate = learning_rate
        
        self.compiled = True
    
    def evaluate(self, X, y):
        """
        Evaluates the model on the given samples, returns the loss
        """

        if not self.compiled:
            raise Exception("You must run compile() at least once before using evaluate()")
            
        y_pred = self.call(X)
        loss = self.loss(y, y_pred)

        # calculate other metrics
        metrics = [metric(y, y_pred) for metric in self.metrics]

        s = f"loss={loss:.4f}"
        for i, metric in enumerate(metrics):
            s += f"\t{self.metrics[i].name}={metric:.4f}"
        print(s)

        return loss, metrics
    
    def fit_step(self, X, y):
        """
        Performs a single training step using the given samples
        """

        # forward propagation
        a = X
        activations = [X]
        for layer in self.layers:
            a = layer(a, cache=True)
            activations.append(a)
        
        # backward propagation
        dL_da = self.loss.derivative(y, a)
        
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            
            # updates the layer's parameters and computes the gradient with respect to the previous layer's activation
            # note: activations[i] is the activation of the previous layer
            dL_dw, dL_db, dL_da = layer.compute_gradients(dL_da, activations[i])
            layer.update_params(dL_dw, dL_db, self.learning_rate)
        
        y_pred = self.call(X)
        loss = self.loss(y, y_pred)

        # calculate other metrics
        metrics = [metric(y, y_pred) for metric in self.metrics]

        return loss, metrics
    
    def fit(self, X, y, epochs=10, verbose=True):
        """
        Trains the model for a fixed number of epochs
        """

        for epoch in range(1, epochs+1):
            loss, metrics = self.fit_step(X, y)
            
            if verbose:
                s = f"loss={loss:.4f}"
                for i, metric in enumerate(metrics):
                    s += f"\t{self.metrics[i].name}={metric:.4f}"
                
                print(f"Epoch {epoch}:\t{s}")
    
    def predict(self, X):
        """
        Generates output predictions for the given input samples
        """

        return self.call(X)
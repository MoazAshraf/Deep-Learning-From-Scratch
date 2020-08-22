from losses import LOSSES, LOSS_DERIVS


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
    
    def compile(self, loss, learning_rate=0.01):
        """
        Configures the model for training, evaluation and/or prediction
        """

        if type(loss) is str:
            self.loss = LOSSES[loss]
            self.loss_deriv = LOSS_DERIVS[loss]
        else:
            # TODO
            pass
        
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

        # TODO: calculate other metrics

        return loss
    
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
        dL_da = self.loss_deriv(y, a)
        
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            
            # updates the layer's parameters and computes the gradient with respect to the previous layer's activation
            # note: activations[i] is the activation of the previous layer
            dL_dw, dL_db, dL_da = layer.compute_gradients(dL_da, activations[i])
            layer.update_params(dL_dw, dL_db, self.learning_rate)
        
        loss = self.loss(y, self.call(X))

        return loss
    
    def fit(self, X, y, epochs=10, verbose=True):
        """
        Trains the model for a fixed number of epochs
        """

        for epoch in range(1, epochs+1):
            loss = self.fit_step(X, y)
            
            if verbose:
                print(f"Epoch {epoch}:\tloss={loss:.4f}")
    
    def predict(self, X):
        """
        Generates output predictions for the given input samples
        """

        return self.call(X)
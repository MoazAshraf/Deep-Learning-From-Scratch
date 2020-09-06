# Deep Learning From Scratch
In this project, I attempt to implement deep learning algorithms from scratch. The
purpose of this is to make sure I understand the theory behind deep learning. And
personally, it's very rewarding to build things from the ground up.

The structure of the library is inspired by PyTorch and Keras. Although I tried to use
vectorization wherever possible, my code is not the most efficient in the world (surprise!)
but it still gets the job done (at least if you're training on MNIST). Also,
there's no GPU support or use of threading, at least for the moment.

Here's an example of how to create and train a model:

```python
from models import Sequential
from layers import Linear, ReLU, Softmax
from losses import CategoricalCrossentropy
from metrics import categorical_accuracy
from optimizers import SGD

# create the model
model = Sequential()
model.add(Linear(128, input_shape=(784,)))
model.add(ReLU())
model.add(Linear(64))
model.add(ReLU())
model.add(Linear(10))
model.add(Softmax())

# configure the model for training
model.configure(loss=CategoricalCrossentropy(),
                optimizer=SGD(learning_rate=0.01),
                metrics={"acc": categorical_accuracy})

# train the model
history = model.train(X_train, y_train, X_test, y_test, epochs=100)
```

```
Epoch 01	loss=1.083168	val_loss=1.117989	acc=0.753333	val_acc=0.720000
...
Epoch 100	loss=0.017159	val_loss=0.237937	acc=0.999000	val_acc=0.939000
```

You can find more examples in the demos folder.

## Features:
- Sequential model
- Layers:
  - Linear (fully connected) layer
  - ReLU
  - Tanh
  - Sigmoid
  - Softmax
  - Dropout
  - Batch Normalization
- Loss Functions:
  - Mean Squared Error (MSE)
  - Binary cross-entropy
  - Categorical cross-entropy
- Metrics:
  - Binary Accuracy
  - Categorical Accuracy
- Regularizers:
  - L1
  - L2
  - L1-L2
- Initializers:
  - Random Normal
  - He Normal
- Optimizers:
  - SGD
  - SGD with Momentum
  - RMSprop
  - Adam

## Dependencies:
The only real dependency is [NumPy](https://numpy.org/). However, I use other
libraries such as Scikit-Learn in the demos because they are convenient.
You can view or install all the dependencies using the [requirements.txt](requirements.txt) file.
For example in pip:

```
pip install -r requirements.txt
```

## License:
MIT License, check the [LICENSE](LICENSE) file.
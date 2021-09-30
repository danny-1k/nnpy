# nnpy

A neural network library written from scratch in numpy 

## Getting started
```
pip install numpy
```


## Example
Training a simple Linear classifier
```python
from sklearn.datasets import make_classification

import nnpy.nn as nn
import nnpy.optim as optim

X,Y = make_classification(n_samples=100, n_features=20, n_informative=10,n_classes=2,shuffle=True, random_state=42)

net = nn.Sequential([
    nn.Linear(20,10),
    nn.ReLU(),
    nn.Linear(10,2),
    nn.Softmax(),
])

loss_fn = nn.CrossEntropyLoss(net)

optimizer = optim.SGD(net,lr=1e-2,)

epochs = 100

for epoch in range(epochs):
    loss_b = []
    for xb,yb in zip(X,Y):
        optimizer.zero_grad()
        xb = np.array([xb])
        pred = net(xb)
        loss = loss_fn(pred,np.array([yb]))
        loss_b.append(loss)
        loss_fn.backward()
        optimizer.step()
    loss = np.array(loss_b).mean()

    acc = 0
    for xb,yb in zip(X,Y):
        optimizer.zero_grad()
        xb = np.array([xb])
        pred = net(xb)
        acc+=sum(pred.argmax(axis=1) == yb)
    acc = acc/len(Y)

    print(f'Epoch {epoch} Loss {loss:3f} Accuracy : {acc}')

```

## Layers

```python

import nnpy.nn as nn

num_inputs  = 2
num_outs = 4
linear_layer = nn.Linear(in_=num_inputs,out_=num_outs)

seq_net = nn.Sequential([
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,4),
])

```

## Activation Layers

Sigmoid, ReLU, Tanh, Softmax

```python
import nnpy.nn as nn
import numpy as np

sig_nonlin = nn.Sigmoid()
tanh_nonlin = nn.Tanh()
relu_nonlin = nn.ReLU()
soft_nonlin = nn.Softmax()

print(sig_nonlin(0))
#0.5

print(tanh_nonlin(np.array([1,2,3])))
#[0.7616, 0.9640, 0.9951]

print(relu_nonlin(np.array([-1,-2,3])))
#[0, 0, 3]

print(soft_nonlin(np.array([1,2,3])))
#[0.0900, 0.2447, 0.6652]
```

## optimizers & lrschedulers

```python
import nnpy.nn as nn

net = nn.Linear(2,2)

optimizer = nn.optim.SGD(net,lr=1e-3)

lrsched = nn.optim.lrschedulers.StepWiseDecay(optimizer)

```

## loss functions

```python
import nnpy.nn as nn
import numpy as np

x = np.array([[1,0],[0,2],[1,3]])
y = np.array([[2,0],[2,2],[1,0]])

net = nn.Linear(2,2)

mseloss = nn.MSE(net)
cross_entropy = nn.CrossEntropy(net)

mseloss(x,y)
#2.3333
cross_entropy(x,y.argmax(axis=1))
#6.1402
```

## Customization
You can create custom functions and Layers by inheriting from `nnpy.core.base.<Class>`
### Example
```python
import nnpy.core.base as base
import numpy as np

class CustomActivationFunc(base.Activation):
    def forward(self,x):
        #make sure to set self.out and self.x
        #and also make sure to set the grads 
        #to zero
        self.x = x
        self.grads['x'] = np.zeros_like(x)
        self.out = np.exp(x)
        return self.out

    def grad_func(self,x):
        #this is the function that will be used
        #to calculate the derivatives in the
        #backward pass
        return np.exp(x)

class SuperFancyLayer(base.Layer):
    def __init__(self):
        self.grads = {'superfancygrad':None}
    def forward(self,x):
        #make sure to set self.out and self.x
        #and also make sure to set the grads 
        #to zero
        self.x = x
        self.grads['superfancygrad'] = np.zeros_like(x)
        self.out = x**2
        return self.out

    def backward(self,grad):
        #this function calculates the grads of
        #its parameters and returns the
        #gradients for the next layer
        self.grads['superfancygrad'] = grad*2*self.x
        return self.grads['superfancygrad']

```

## MNIST demo


To train the model, run `python mnist/train.py`


To test the model on numbers drawn by me(Bad handwriting lol) run `python mnist/test.py` to test 


## nnpy
- core
- nn
- optim

## nnpy.core
- base

## nnpy.nn:
- activations
- layers
- loss

## nnpy.optim:
- lrschedulers
- optim

## nnpy.nn.activations:
- ReLU
- Sigmoid
- Tanh
- Softmax

## nnpy.nn.layers:
- Linear
- Dropout
- Sequential
- TimeDistributed

## nnpy.nn.loss:
- CrossEntropy
- MSE
- MAE

## nnpy.optim.lrschedulers
- StepWiseDecay
- ReduceOnPlateau

## nnpy.optim.optim
- SGD
- Adagrad

## TODO
- [ ] Adam optimizer
- [ ] CNN
- [ ] RNN
- [ ] LSTM
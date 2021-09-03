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

## Customization
You can create custom functions and Layers by inheriting from `nnpy.core.base.<Class>`
### Example
```python
import nnpy.core.base as base
impor numpy as np

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
        #its parameters and returns a 
        #gradient for the next layer to backward
        self.grads['superfancygrad'] = grad*2*self.x
        return self.grads['superfancygrad']

```

## MNIST demo


To train the model, run `python mnist/train.py`


To test the model on numbers drawn by me(Bad handwriting lol) run `python mnist/test.py` to test 

## TODO
- [ ] Adam optimizer
- [ ] CNN
- [ ] RNN
import numpy as np
from ..core.base import Layer,Function
from .activations import Tanh,Softmax

class Linear(Layer):
    def __init__(self,in_,out_):
        super().__init__()
        self.in_dims = in_
        self.out_dims = out_
        self.params = {'w':np.random.randn(in_,out_)*np.sqrt(1/out_),
                        'b':np.random.randn(1,out_)*np.sqrt(1/out_),
        }
        self.grads = {'w':np.zeros_like(self.params['w']),'b':np.zeros_like(self.params['b'])}
    def forward(self,x):
        '''
        Performs computation : x@w+b
        '''
        self.x = x
        self.out = (x@self.params['w'])+self.params['b']
        return self.out
    
    def backward(self,grad):
        self.grads['w'] += self.x.T@grad
        self.grads['b'] += grad.sum(axis=0)
        return grad@self.params['w'].T

    def step(self,lr):
        'Updates layer parameters'
        self.params['w'] -= lr*self.grads['w']
        self.params['b'] -= lr*self.grads['b']

class Dropout(Function):
    '''
    Drops out (sets to zero) a percentage of 
    the input
    It's an effective method of combating overfitting 
    '''
    def __init__(self,drop_p):
        super().__init__()
        self.drop_p = drop_p

    def func(self,x):
        if not self.eval:
            keep_prob = 1-self.drop_p
            mask = np.random.uniform(0,1,x.shape)<keep_prob
            scale = 1/keep_prob if keep_prob > 0 else 0
            return mask * x * scale 
        else:
            return x

class Sequential(Layer):
    def __init__(self,layers):
        self.layers = layers
    def forward(self,x):
        self.x = x
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def zero_grad(self):
        '''
        Zeros the grads of all layers in the net
        if the layer isn't a Function
        '''
        for layer in self.layers:
            if not isinstance(layer,Function):
                layer.zero_grad()
    def backward(self,grad):
        '''
        Push gradients through the layers
        '''
        for layer in self.layers[::-1]:
            if not isinstance(layer,Function):
                grad = layer.backward(grad)

    def eval(self):
        '''
        Sets the network in eval mode
        '''
        for layer in self.layers:
            if isinstance(layer,Function):
                layer.eval = True
    
    def train(self):
        '''
        Sets the network in train mode
        '''
        for layer in self.layers:
            if isinstance(layer,Function):
                layer.eval = False
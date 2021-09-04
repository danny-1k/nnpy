import numpy as np
from ..core.base import Activation

class ReLU(Activation):
    '''
    Rectified Linear Unit is an activation function
    that clips off negative values to zero
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        '''
        Computes the function f(x) = max(x,0)
        on the input
        '''
        self.grads['x'] = np.zeros_like(x)
        self.x = x
        self.out = np.maximum(x,0)
        return self.out

    def grad_func(self,x):
        '''
        The derivative of the ReLU activation with respect
        to its input
        '''
        return 1*(x>0)

class Sigmoid(Activation):
    '''
    The sigmoid activation function squashes its inputs
    between zero and one
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        '''
        Computes the function f(x) = 1/(1+e^-x)
        on the input
        '''
        x = np.clip(x,-500,500)
        self.grads['x'] = np.zeros_like(x)
        self.x = x
        self.out = 1/(1+np.exp(-x))
        return self.out

    def grad_func(self,x):
        '''
        The derivative of the sigmoid activation with respect
        to its input
        '''
        return x*(1-x)

class Tanh(Activation):
    '''
    Hyperbolic Tangent activation function squashes its input
    between -1 and 1
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        '''
        Computes the function f(x) = (e^x-e^-x)/(e^x+e^-x)
        on the input
        '''
        x = np.clip(x,-500,500)
        self.grads['x'] = np.zeros_like(x)
        self.x = x
        self.out = np.tanh(x)
        return self.out

    def grad_func(self,x):
        '''
        The derivative of the tanh activation with respect
        to its input
        '''
        return 1-(x**2)

class Softmax(Activation):
    '''
    Softmax activation function converts raw logits
    to probabilities 
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        '''
        Computes the function f(x) = e**x/sum(e**x)
        on the input
        '''
        x = np.clip(x,-500,500)
        self.grads['x'] = np.zeros_like(x)
        self.x = x
        exps = np.exp(self.x)
        self.out = exps/np.sum(exps,axis=1,keepdims=True)
        return self.out

    def grad_func(self,x):
        '''
        The derivative of the softmax activation with respect
        to its input
        '''
        return x*(1-x)
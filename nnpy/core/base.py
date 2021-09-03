import numpy as np

class Layer:
    '''
    Layer base class
    '''
    def __init__(self):
        self.out = None
        self.grads = {}
        self.x = None
    def forward(self,x):
        '''
        Forward method to push inputs through a computation
        and then forward through the rest of the network
        '''
        pass

    def backward(self,grad):
        '''
        Backward method takes incoming gradients to calculate
        the gradients of the parameters
        '''
        pass

    def __call__(self,*args):
        return self.forward(*args)

    def __repr__(self):
        return f'{type(self).__name__}()'
        
    def zero_grad(self):
        '''
        Method to zero the gradients of all parameters
        '''
        for item in self.grads:
            self.grads[item] = np.zeros_like(self.grads[item])

class Function:
    '''
    Function base class

    A class that inherits this class can push 
    inputs forward but can't push gradients backward
    '''
    def __init__(self):
        self.grads = {'x':None}
        self.eval = False
    def forward(self,x):
        '''
        Performs self.func on x
        '''
        return self.func(x)

    def func(self,x):
        '''
        Method returns an operation that should be done on x
        '''
        pass

    def __call__(self,*args):
        return self.forward(*args)

    def __repr__(self):
        return f'{type(self).__name__}()'


class Activation(Layer):
    '''
    Activation base class
    '''
    def __init__(self):
        super().__init__()
        self.grads = {'x': None}
    def grad_func(self, x):
        pass
    def backward(self,grad):
        self.grads['x'] += self.grad_func(self.out)*grad
        return self.grads['x']

class Loss(Layer):
    '''
    Loss base class
    '''
    def __init__(self,net):
        self.net = net
        self.grads = {'x':None}

    def forward(self,pred,targets):
        pass

    def backward(self):
        self.grads['x'] = self.grad_func(self.pred,self.targets)
        self.net.backward(self.grads['x'])

    def grad_func(self,pred,targets):
        pass

class Optim:
    '''
    Optimizer base class
    '''
    def __init__(self,net,lr):
        self.lr = lr or 1e-2
        self.net = net

    def step(self):
        '''
        Method for updating parameters of the network
        '''
        pass
    
    def zero_grad(self):
        '''
        Calls .zero_grad() on all the layers of the network
        if the layer is not a function
        '''
        for layer in self.net.layers:
            if not isinstance(layer,Function):
                layer.zero_grad()


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.iter = 0

    def reduce_lr(self, lr):
        pass

    def set_lr(self, lr):
        self.optimizer.lr = lr

    def step(self):
        '''
        calls the reduction algorithm on the optimizer
        '''
        self.reduce_lr(self.optimizer.lr)
        self.iter += 1

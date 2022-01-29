import numpy as np
import pickle

class Layer:
    '''
    Layer base class
    '''
    def __init__(self):
        self.out = None
        self.grads = {}
        self.x = None
    def forward(self,*args,**kwargs):
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


    def return_state_dict(self,):
        state_dict = {}
        
        if 'params' in dir(self):
            state_dict['params'] = self.params

        return state_dict


    def load_state_dict(self,state_dict):
        for key in state_dict:
            if key in dir(self):
                self.__setattr__(key,state_dict[key])
            
            else:
                raise ValueError(f'{key} is not an attribute of this object.')


    def save_state_dict(self,f):
        pickle.dump(self.return_state_dict(),open(f,'wb'))


    def __call__(self,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs
        return self.forward(*args,**kwargs)

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
        self.x = x
        self.out = self.func(x)
        return self.out

    def func(self,x):
        '''
        Method returns an operation that should be done on x
        '''
        pass

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def __repr__(self):
        return f'{type(self).__name__}()'


    def return_state_dict(self,):
        return {}


    def load_state_dict(self,state_dict):
        for key in state_dict:
            if key in dir(self):
                self.__setattr__(key,state_dict[key])
            else:
                raise ValueError(f'{key} is not an attribute of this object.')

    def save_state_dict(self,f):
        pickle.dump(self.return_state_dict(),open(f,'wb'))


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
        if 'layers' not in dir(self.net):
            return
        for layer in self.net.layers:
            if not isinstance(layer,Function):
                layer.zero_grad()


    def return_state_dict(self,):
        state_dict = {}
        
        if 'params' in dir(self):
            state_dict['params'] = self.params

        return state_dict

    
    def load_state_dict(self,state_dict):
        for key in state_dict:
            self.__setattr__(key,state_dict[key])


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

    
    def return_state_dict(self,):
        state_dict = {}
        
        if 'params' in dir(self):
            state_dict['params'] = self.params

        if 'iter' in dir(self):
            state_dict['iter'] = self.iter

        if 'lr' in dir(self):
            state_dict['lr'] = self.lr


        return state_dict
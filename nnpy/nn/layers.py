import numpy as np
from ..core.base import Layer,Function
from ..core.utils import create_graph
from .activations import Tanh,Softmax

class Linear(Layer):
    def __init__(self,in_,out_):
        super().__init__()
        self.in_dims = in_
        self.out_dims = out_
        self.params = {'w':np.random.uniform(-np.sqrt(1/out_),np.sqrt(1/out_),(in_,out_)),
                        'b':np.random.uniform(-np.sqrt(1/out_),np.sqrt(1/out_),(1,out_)),
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

        return grad

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

class Module(Layer):


    def forward(self,*args,**kwargs):
        pass

    def backward(self, grad):
        if 'layers' not in dir(self):
            pass

        for layer in self.layers[::-1]:
            if not isinstance(layer,Function):
                grad = layer.backward(grad)

        
        return grad


    def zero_grad(self):
        '''
        Zeros the grads of all layers in the net
        if the layer isn't a Function
        '''
        
        if 'layers' not in dir(self):
            pass

        for layer in self.layers:
            if not isinstance(layer,Function):
                layer.zero_grad()


    def eval(self):
        '''
        Sets the network in eval mode
        '''

        if 'layers' not in dir(self):
            pass

        for layer in self.layers:
            if isinstance(layer,Function):
                layer.eval = True


    def train(self):
        '''
        Sets the network in train mode
        '''

        if 'layers' not in dir(self):
            pass


        for layer in self.layers:
            if isinstance(layer,Function):
                layer.eval = False


    def __call__(self, *args,**kwargs):
        self.out = self.forward(*args,**kwargs)

        if 'layers' not in dir(self):
            x = [*args]
            layers = []
            
            for item in dir(self):
                if isinstance(self.__getattribute__(item),Layer) or isinstance(self.__getattribute__(item),Function):
                    layers.append(self.__getattribute__(item))


            for xi in x:
                graph = create_graph(xi,layers)
                if len(graph) >0:
                    layers = graph

            self.layers = layers

        return self.out


class TimeDistributed(Layer):
    def __init__(self, layer):
        self.layer = layer
        self.grads = layer.grads
        self.params = layer.params if 'params' in dir(layer) else None

    def forward(self, x):
        self.params = self.layer.params if 'params' in dir(
            self.layer) else None
        # x is expected to be of shape (batch_size,seq_len,in_dims)
        self.x = x
        self.out = []
        time_steps = x.shape[1]
        for t in range(time_steps):
            p = self.layer(self.x[:, t, :])
            self.out.append(p)
        # out is now of shape (seq_len,batch_size,in_dims)
        # should be (batch_size,seq_len,in_dims)
        self.out = np.array(self.out).transpose(1, 0, 2)
        return self.out

    def backward(self, grad):
        self.params = self.layer.params if 'params' in dir(
            self.layer) else None
        # grad is expected to be of shape (batch_size,seq_len,in_dims)
        time_steps = self.x.shape[1]
        next_grad = np.zeros_like(self.x)

        for t in reversed(range(time_steps)):
            x = self.x[:, t, :]
            self.layer.x = x
            next_grad[:,t,:] = self.layer.backward(grad[:, t, :])
        
        self.grads = {item:self.layer.grads[item] for item in self.layer.grads}
        return next_grad

    def step(self, lr):
        self.layer.step(lr)

    def __repr__(self):
        return f'TimeDistributedLayer({self.layer})'
import numpy as np
from ..core.base import Optim,Function

class SGD(Optim):
    def __init__(self,net,lr,momentum=None):
        '''
        Stochastic gradient descent updates the parameters
        of the network by multiplying the gradients by 
        a learning rate
        '''
        super().__init__(net,lr)
        self.momentum = momentum
        #Momentum algorithm enhances the optimization process by
        #accumulating exponentially decaying past gradients 
        #you can picture a ball rolling down a hill
        if momentum:
            self.v = {}
            for idx,layer in enumerate(self.net.layers):
                val = {}
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    for item in layer.grads:
                        val[item] = np.zeros_like(layer.grads[item])
                    self.v[idx] = val

    def step(self):
        if self.momentum:
            for idx,layer in enumerate(self.net.layers):
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    for item in layer.grads:
                        self.v[idx][item] = self.momentum * \
                            self.v[idx][item] + (self.lr * layer.grads[item])
                        layer.params[item] -= self.v[idx][item]
        else:
            for layer in self.net.layers:
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    layer.step(self.lr)


class Adagrad(Optim):
    '''
        Adaptive Gradient Algorithm works by making each parameter 
        in the neural network have it's own learning rate
        '''
    def __init__(self,net,lr=1e-4,epsilon=1e-8):
        super().__init__(net,lr)
        self.epsilon = epsilon
        self.g = {}
        self.val = {}
        for idx,layer in enumerate(self.net.layers):
            val = {}
            if not 'x' in layer.grads and not isinstance(layer,Function) :
                for item in layer.grads:
                    val[item] = np.zeros_like(layer.params[item])
            self.val[idx] = val
    
    def step(self):
        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function) :
                for item in layer.grads:
                    self.val[idx][item]+=layer.grads[item]**2
        
        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function) :
                for item in layer.grads:
                    layer.params[item] -= layer.grads[item] * \
                        (self.lr/(np.sqrt(self.val[idx][item])+self.epsilon))
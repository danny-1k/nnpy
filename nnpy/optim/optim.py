import numpy as np
from ..core.base import Optim,Function

class SGD(Optim):
    def __init__(self,net,lr=1e-2,momentum=None):
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
    
    def step(self):

        #when first self.step() is called the first time,
        #instantiate all necessary vals
        
        if self.g=={} and self.val == {}:
            for idx,layer in enumerate(self.net.layers):
                val = {}
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    for item in layer.grads:
                        val[item] = np.zeros_like(layer.params[item])
                self.val[idx] = val

        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function) :
                for item in layer.grads:
                    self.val[idx][item]+=layer.grads[item]**2
        
        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function) :
                for item in layer.grads:
                    layer.params[item] -= layer.grads[item] * \
                        (self.lr/(np.sqrt(self.val[idx][item])+self.epsilon))

class RMSprop(Optim):
    def __init__(self, net, lr=1e-4,beta=.9,epsilon=1e-8):
        self.v = {}
        self.beta = beta
        self.epsilon = epsilon
        self.lr = lr
        self.net = net

    def step(self):
        if self.v == {}:
            for idx,layer in enumerate(self.net.layers):
                v = {}
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    for item in layer.grads:
                        v[item] = np.zeros_like(layer.params[item])
                self.v[idx] = v

        
        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function):
                for item in layer.grads:
                    self.v[idx][item] += self.beta*self.v[idx][item] + (1-self.beta) * (layer.grads[item]**2)

        for idx,layer in enumerate(self.net.layers):
            if not 'x' in layer.grads and not isinstance(layer,Function):
                for item in layer.grads:
                    layer.params[item] -= self.lr/((self.v[idx][item]+1e-8)**.5) * layer.grads[item]


class Adam(Optim):
    def __init__(self,net,lr=1e-4,beta1=0.9,beta2=0.999,epsilon=1e-8):
        super().__init__(net, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self):
        self.t+=1 #increment t

        if self.m == {} and self.v == {}:
            for idx,layer in enumerate(self.net.layers):
                m = {}
                v = {}
                
                if not 'x' in layer.grads and not isinstance(layer,Function) :
                    for item in layer.grads:
                        m[item] = np.zeros_like(layer.params[item])
                        v[item] = np.zeros_like(layer.params[item])
                        
                self.m[idx] = m
                self.v[idx] = v
                

        for idx,layer in enumerate(self.net.layers):

            if not 'x' in layer.grads and not isinstance(layer,Function):
                for item in layer.grads:

                    self.m[idx][item] = self.beta1 * self.m[idx][item] + (1-self.beta1) * layer.grads[item]
                    self.v[idx][item] = self.beta2 * self.v[idx][item] + (1-self.beta2) * pow(layer.grads[item],2)

                    m_corrected = self.m[idx][item]/(1-pow(self.beta1,self.t))
                    v_corrected = self.v[idx][item]/(1-pow(self.beta2,self.t))


                    layer.params[item] -= m_corrected*(self.lr/(pow(v_corrected,.5)+self.epsilon))
import numpy as np
from ..core.base import Loss

class CrossEntropy(Loss):
    '''
    Cross Entopy Loss punishes the net for having a low
    confidence in it's outputs
    Example:
    softmax(x)-> 0.2; high loss
    softmax(x) ->0.9; low cross entropy loss
    '''
    def __init__(self,net):
        super().__init__(net)

    def forward(self,pred,targets):
        '''
        Calculates nl(pred[target])
        '''
        #pred should be of shape (batch_size,num_classes)
        #targets should be of shape (batch_size)
        N = pred.shape[0]
        self.pred = pred
        self.targets = targets
        #adding 1e-8 to prevent taking the log of 0!
        self.out = np.sum(-np.log(pred[range(N),targets]+1e-8))*(1/N)
        return self.out

    def grad_func(self,pred,targets):
        '''
        Returns the grad of the pred; pred-targets
        '''
        N = pred.shape[1]   
        grad = pred-(np.eye(N)[targets])
        return grad

class MSE(Loss):
    '''
    MSE is the Mean Squared Error of the predictions
    '''
    def __init__(self,net):
        super().__init__(net)

    def forward(self,pred,targets):
        '''
        Calculates (1/N)*(pred-targets)**2
        '''
        self.pred = pred
        self.targets = targets
        self.out = (1/(pred.shape[0]*pred.shape[1]))*np.sum((pred-targets)**2)
        return self.out

    def grad_func(self,pred,targets):
        '''
        Returns the grad of the pred; 2*(pred-targets)
        '''
        return 2*(pred-targets)

class MAE(Loss):
    '''
    MAE is the Mean Absolute Error of the predictions
    '''
    def __init__(self,net):
        super().__init__(net)

    def forward(self,pred,targets):
        '''
        Calculates (1/N)*abs(pred-targets)
        '''
        self.pred = pred
        self.targets = targets
        self.out = (1/(pred.shape[0]*pred.shape[1]))*np.sum(abs(pred-targets))
        return self.out

    def grad_func(self,pred,targets):
        '''
        Returns the grad of the pred; (pred-targets)>0
        '''
        return ((pred-targets) > 0) + 0
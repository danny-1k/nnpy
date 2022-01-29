from ..core.base import LRScheduler

class StepWiseDecay(LRScheduler):
    '''
    Reduces the learning rate by some factor, gamma
    after a numer of iterations
    '''
    def __init__(self, optimizer, step_size=1000, gamma=0.5,min_lr=1e-4):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

    def reduce_lr(self, lr):
        if ((self.iter+1) % self.step_size) == 0:
            lr = self.optimizer.lr * self.gamma
            if lr < self.min_lr:
                lr = self.min_lr    
            self.set_lr(lr)

class ReduceOnPlateau(LRScheduler):
    '''
    Reduces the learning rate by some factor, gamma
    after there hasn't been any reduction in loss after 
    a certain period
    '''
    def __init__(self, optimizer, loss_class, patience=5, gamma=0.5,min_lr=1e-4):
        super().__init__(optimizer)
        self.patience = patience
        self.gamma = gamma
        self.min_lr = min_lr
        self.loss_class = loss_class
        self.loss_history = []

    def reduce_lr(self, lr):
        if len(self.loss_history) == self.patience and not(self.loss_history[-1] < self.loss_history[0]):
            lr = self.optimizer.lr * self.gamma

            if lr < self.min_lr:
                lr = self.min_lr   
            
            self.set_lr(lr)
            self.loss_history = []
        self.loss_history.append(self.loss_class.out)

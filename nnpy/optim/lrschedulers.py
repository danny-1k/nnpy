from ..core.base import LRScheduler

class StepWiseDecay(LRScheduler):
    '''
    Reduces the learning rate by some factor, gamma
    after a numer of iterations
    '''
    def __init__(self, optimizer, step_size=1, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def reduce_lr(self, lr):
        if (self.iter % self.step_size) == 0:
            lr = self.optimizer.lr * self.gamma
            self.set_lr(lr)

class ReduceOnPlateau(LRScheduler):
    '''
    Reduces the learning rate by some factor, gamma
    after there hasn't been any reduction in loss after 
    a certain period
    '''
    def __init__(self, optimizer, loss_class, patience=5, gamma=0.5):
        super().__init__(optimizer)
        self.patience = patience
        self.gamma = gamma
        self.loss_class = loss_class
        self.loss_history = []

    def reduce_lr(self, lr):
        if len(self.loss_history) == self.patience and not(self.loss_history[-1] < self.loss_history[0]):
            lr = self.optimizer.lr * self.gamma
            self.set_lr(lr)
            self.loss_history = []
        self.loss_history.append(self.loss_class.out)

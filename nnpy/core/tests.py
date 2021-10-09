import numpy as np
from numpy.linalg import norm


def numerical_grad(func,x):
    '''
    Function to calculate gradients
    numerically

    f(x+h) - f(x-h)
    ---------------
           2h

    '''

    
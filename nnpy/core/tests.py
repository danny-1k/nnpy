import numpy as np
from numpy.linalg import norm


def numerical_grad(func,x):
    '''
    Function to calculate gradients
    of a function numerically

    f(x+h) - f(x-h)
    ---------------
           2h

    '''

    h = 1e-7

    grad_vec = []

    #assuming x is a (num,shape) matrix
    #for example (2,3,3) matrix
    #then we need to compute the gradient one at a time

    if len(x.shape) > 2:
        for xi in x:
            f_x_plus_h = func(xi+h)
            f_x_min_h = func(xi-h)

            g = (f_x_plus_h-f_x_min_h)/(2*h)

            grad_vec.append(g)

        grad_vec =  np.array(grad_vec).T

    else:
        f_x_plus_h = func(x+h)
        f_x_min_h = func(x-h)

        grad_vec = (f_x_plus_h-f_x_min_h)/(2*h)

    
    return grad_vec


def compare(num_grads,grads):
    diff = abs(num_grads-grads)
    bools = diff < 1e-8
    if np.sum(bools) == bools.size:
        print('The derivated gradients are correct')
    else:
        print('The derivated gradients are wrong.')
    return diff


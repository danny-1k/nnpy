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


def check_grads_layer(layer,x,deriv_grads):
    '''
    Checks the grads of the params of layers
    '''
    h = 1e-7
    for param in layer.params:
        orig_param = layer.params[param]
        #layer.params[param] = orig_param+h
        #f_x_plus_h = layer(x)

        #layer.params[param] = orig_param-h
        #f_x_min_h = layer(x)

        #layer.params[param] = orig_param

        #g = (f_x_plus_h-f_x_min_h)/(2*h)
        g = ((orig_param+h)-(orig_param-h))/(2*h)
        if np.sum(abs(g-deriv_grads[param]) < 1e-7) == g.size:
            print(f'{param} grads -> Correct')
        else:
            print(f'{param} grads -> Wrong')
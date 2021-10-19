import numpy as np
from numpy.linalg import norm


def numerical_grad(func,x):
    '''
    Function to calculate gradients
    of a function with a single parameter numerically

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

def numerical_grad_layer(layer,lossfn,x,y,param_name,layer_num=None):
    '''
    Function to calculate gradients
    of a layer with respect to a parameter numerically

    layer(x,param+h) - layer(x,param-h)
    -----------------------------------
                     2*h 

    the numerical gradients are calculated differently in that
    every value in the parameter is tweaked and then we measure the 
    effect on the whole layer.
    The loss is then calculated when we add and when we subtract

    So it would be like this:
    assuming:
    `param` = [1,2,3]
    `x` = [3,4,5]
    `y` = [6,7,8]
    loss(p,y) = sum(p-y)
    `h` = 1e-4
    net(param) = x+param

    The numerical grad of param would be calculated as

    [ 
        (loss(net([1+h,2,3]),y) - loss(net([1-h,2,3]),y)) / 2*h,

        (loss(net([1,2+h,3]),y) - loss(net([1,2-h,3]),y)) / 2*h,

        (loss(net([1,2,3+h]),y) - loss(net([1,2,3-h]),y)) / 2*h
    ]
    
    
    The result is [0.9999999999976694, 0.9999999999976694, 0.9999999999976694]
    
    which is pretty close to [1,1,1]

    '''

    if layer_num:
        try:
            layer = layer.layers[layer_num]
        except:
            pass

    orig_param = layer.params[param_name].copy()
    h_vec = np.zeros(np.prod(orig_param.shape))
    n_grad = np.zeros_like(h_vec)

    h = 1e-7

    for idx in range(np.prod(orig_param.shape)):
        h_vec[idx] = h

        layer.params[param_name] = orig_param + h_vec.reshape(orig_param.shape)
        l1 = lossfn(layer(x),y)

        layer.params[param_name] = orig_param - h_vec.reshape(orig_param.shape)
        l2 = lossfn(layer(x),y)

        n_grad[idx] = (l1-l2)/(2*h)

        h_vec[idx] = 0

    
    n_grad = n_grad.reshape(orig_param.shape)

    return n_grad
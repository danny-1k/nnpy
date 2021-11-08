import math
import numpy as np

def create_graph(x,layers):
    idx = 0
    g = []
    while True:
        try:
            layer = layers[idx]
            if layer.x is not None:
                if layer.x.data == x.data:
                    g.append(layer)
                    x = layer.out
                    del layers[idx]
                    idx = 0
                else:
                    if idx == len(layers):
                        idx = 0
                    else:
                        idx+=1

                if len(layers) == 0:
                    break
            else:
                break
        except:
            
            break

    return g


def add_padding(x,padding=0):

    original_shape = x.shape #x should be of shape (batch_size,height,width)
    
    assert len(original_shape) == 3, f"Expected input to be of shape (batch_size,height,width). Got {original_shape} instead"


    for i in range(padding):
        x = np.insert(x,0,np.zeros((x.shape[0],x.shape[-1])),axis=1)#zeros on the top
        x = np.insert(x,0,np.zeros((x.shape[0],x.shape[-1])),axis=2)#zeros on the left
        
        x = np.insert(x,x.shape[-1],np.zeros((x.shape[0],original_shape[-1])),axis=2)#zeros on the right
        x = np.insert(x,original_shape[-1],np.zeros((x.shape[0],x.shape[-1])),axis=1)#zeros on the bottom
        original_shape = x.shape

    return x


def gen_patches(x,kernel_size,stride=(1,1),padding=0):
    #x should be of shape (batch_size,height,width)
    assert len(x.shape) == 3, f"Expected input to be of shape (batch_size,height,width). Got {x.shape} instead"
    assert len(stride) == 2,f"Expected stride to be a tuple of length 2"
    w = x.shape[-1]
    h = x.shape [-2]

    x = add_padding(x,padding)

    assert w>=kernel_size[0] and h>=kernel_size[1],f"input shouldn't be smaller than the kernel_size. x, {(w,h)} kernel, {kernel_size}." 

    out_h = ((h-kernel_size[0]+2*padding)/stride[0])+1
    out_w = ((w-kernel_size[1]+2*padding)/stride[1])+1

    out_h = math.floor(out_h+.5) #rounding to the nearest whole number
    out_w = math.floor(out_w+.5) #rounding to the nearest whole number

    out = np.zeros((x.shape[0],out_h,out_w,*kernel_size))

    for i in range(out_h):
        for j in range(out_w):
            patch = x[
                :,i+(i*(stride[0]-1)):kernel_size[0]+i+(i*(stride[0]-1)),\
                    j+(j*(stride[1]-1)):kernel_size[1]+j+(j*(stride[1]-1))
                ]

            out[:,i,j] = patch
    
    return out




def correlation2d(x,kernel,stride=(1,1),padding=0):
    #x should be of shape (batch_size,height,width)
    assert len(x.shape) == 3, f"Expected input to be of shape (batch_size,height,width). Got {x.shape} instead"
    assert len(stride) == 2,f"Expected stride to be a tuple of length 2"
    w = x.shape[-1]
    h = x.shape [-2]
    kernel_size = kernel.shape

    x = add_padding(x,padding)

    #x shouldn't be smaller than the kernel size

    assert w>=kernel_size[0] and h>=kernel_size[1],f"input shouldn't be smaller than the kernel_size. x, {(w,h)} kernel, {kernel_size}." 

    out_h = ((h-kernel_size[0]+2*padding)/stride[0])+1
    out_w = ((w-kernel_size[1]+2*padding)/stride[1])+1

    out_h = math.floor(out_h+.5) #rounding to the nearest whole number
    out_w = math.floor(out_w+.5) #rounding to the nearest whole number

    out = np.zeros((x.shape[0],out_h,out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = x[
                :,i+(i*(stride[0]-1)):kernel_size[0]+i+(i*(stride[0]-1)),\
                    j+(j*(stride[1]-1)):kernel_size[1]+j+(j*(stride[1]-1))
                ]
            
            out[:,i,j] = (patch*kernel).sum(axis=(1,2))
    
    return out


def correlation2d_backward(x,grad,kernel):
    #grad of shape (batch_size,out_height,out_width)
    #kernel of shape (kernel_height,kernel_width)
    #x is the kernel patches of x

    grad = grad.reshape(grad.shape[0],-1) #(batch_size,out_height*out_width)
    kernel_grad = np.zeros(1,np.prod(kernel.shape))
    #kernel_grad of shape (1,kernel_height*kernel_width)

    for patch in x:
        #patch of shape (batch_size,kernel_height,kernel_width)
        #(kernel_height*kernel_width,batch_size) @ (batch_size,out_height*out_width)
        kernel_grad += (patch.reshape(-1,np.prod(kernel.shape)).T @ grad).sum(axis=1)

    return kernel_grad.reshape(kernel.shape)
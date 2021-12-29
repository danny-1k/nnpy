import math
import numpy as np

def create_graph(x,layers):
    #this is the shittiest computational graph maker ever
    #so fucking dumb
    #this is the only way I could think of that could work with the
    #structure of the layers and stuff.
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
    shape = np.array(x.shape)
    shape[-2:] += 2*padding
    out = np.zeros((x.shape[0],x.shape[1],x.shape[2]+(2*padding),x.shape[3]+(2*padding)))

    out[:,:,padding:x.shape[2]+padding,padding:x.shape[3]+padding] = x
    
    return out


def gen_patches(x,kernel_size,stride=(1,1),padding=0):
    '''
    Returns the patches for convolution of the input
    the output is of shape (in_channels,out_h,out_w,kernel_h,kernel_w)
    '''

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

    out = np.zeros((x.shape[0],out_h,out_w,*kernel_size)) #(batch_size,out_height,out_width,kernel_height,kernel_width)

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
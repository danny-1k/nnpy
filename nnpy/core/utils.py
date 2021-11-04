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


def add_padding(x,padding):
    original_shape = x.shape #x should be of shape (batch_size,height,width)
    
    for i in range(padding):
        x = np.insert(x,0,np.zeros((x.shape[0],x.shape[-1])),axis=1)#zeros on the top
        x = np.insert(x,0,np.zeros((x.shape[0],x.shape[-1])),axis=2)#zeros on the left
        x = np.insert(x,x.shape[-1],np.zeros((x.shape[0],original_shape[-1])),axis=2)#zeros on the right
        x = np.insert(x,original_shape[-1],np.zeros((x.shape[0],x.shape[-1])),axis=1)#zeros on the bottom

        original_shape = x.shape

    return x


# def cross_correlation(x,kernel,stride=(1,1),padding=0):
#     w = x.shape[-1]
#     h = x.shape [-2]

#     kernel_size = kernel.shape

#     out_h = ((h-kernel_size[0]+2*padding)/stride[0])+1
#     out_w = ((w-kernel_size[1]+2*padding)/stride[1])+1

    
#     for 
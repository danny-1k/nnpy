import sys
sys.path.append('..')

from PIL import Image
import numpy as np

from model import net

def transform(x,ops):
    for op in ops:
        x = op(x)
    return x

transforms = [
    lambda x:x.convert('L'),
    lambda x:np.asarray(x),
    lambda x:x/255,
    lambda x:x.reshape(1,28*28),
]

w1 = np.load('weights/w1.npy')
w2 = np.load('weights/w2.npy')
b1 = np.load('weights/b1.npy')
b2 = np.load('weights/b2.npy')
w3 = np.load('weights/w3.npy')
b3 = np.load('weights/b3.npy')

net.layers[0].params['w'] = w1
net.layers[0].params['b'] = b1
net.layers[2].params['w'] = w2
net.layers[2].params['b'] = b2
net.layers[4].params['w'] = w3
net.layers[4].params['b'] = b3

net.eval()

labels = ['0','1','2','3','4','5','6','7','8','9']
for i in labels:
    img = transform(Image.open(f'test_imgs/{i}.png'),transforms)
    pred = net(img).squeeze()
    print(f'\t{"$" if str(pred.argmax()) == i else "X"}\t Computer : {pred.argmax()}\tActual : {i}\tConfidence : {pred[pred.argmax()]:.4f}')
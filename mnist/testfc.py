import sys
sys.path.append('..')

from PIL import Image
import numpy as np

from model import FC

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

net = FC()
net.load_weights('FC')
net.eval()

labels = ['0','1','2','3','4','5','6','7','8','9']
for i in labels:
    img = transform(Image.open(f'test_imgs/{i}.png'),transforms)
    pred = net(img).squeeze()
    print(f'\t{"$" if str(pred.argmax()) == i else "X"}\t Computer : {pred.argmax()}\tActual : {i}\tConfidence : {pred[pred.argmax()]:.4f}')
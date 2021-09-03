import sys
sys.path.append('..')

import nnpy.nn as nn
from nnpy.optim import SGD
from model import net
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
trans = transforms.Compose(
    [
        lambda x:np.asarray(x),
        lambda x:x/255,
    ]
)

train = datasets.MNIST('~', train=True, transform=trans)
test = datasets.MNIST('~', train=False, transform=trans)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=True)

epochs = 10
lr = 1e-2

loss_fn = nn.CrossEntropy(net)
optim = SGD(net,lr=lr)
print('Started training!')
test_loss_over_time = []
train_loss_over_time = []
acc_over_time = []
for e in range(epochs):
    train_batch_loss = []
    for x,y in train_loader:
        x = x.view(-1,28*28).numpy()
        y = y.numpy()
        p = net(x)
        loss = loss_fn(p,y)
        train_batch_loss.append(loss)
        loss_fn.backward()
        #print(net.layers[0].grads['w'])
        optim.step()
        optim.zero_grad()
    test_batch_loss = []
    accuracy = 0
    for x, y in test_loader:
        x = x.view(-1,28*28).numpy()
        y = y.numpy()
        pred = net(x)
        loss = loss_fn(pred, y)
        test_batch_loss.append(loss)
        accuracy += sum(pred.argmax(axis=1) == y)
    accuracy = accuracy/len(test)
    acc_over_time.append(accuracy)
    test_loss_over_time.append(np.mean(test_batch_loss))

np.save('weights/w1',net.layers[0].params['w'])
np.save('weights/b1',net.layers[0].params['b'])
np.save('weights/w2',net.layers[2].params['w'])
np.save('weights/b2',net.layers[2].params['b'])
np.save('weights/w3',net.layers[4].params['w'])
np.save('weights/b3',net.layers[4].params['b'])

plt.plot(acc_over_time)
plt.savefig('plots/accuracy.png')
plt.plot(test_loss_over_time)
plt.savefig('plots/loss.png')
plt.close('all')
import sys
sys.path.append('..')

import nnpy.nn as nn
from nnpy.optim import Adam

from model import FC

from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import numpy as np

trans = transforms.Compose(
    [
        lambda x:np.asarray(x),
        lambda x:x/255,
    ]
)

net = FC()

train = datasets.MNIST('~', train=True, transform=trans)
test = datasets.MNIST('~', train=False, transform=trans)

train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=64, shuffle=True)

epochs = 15
lr = 1e-4

lossfn = nn.CrossEntropy(net)
optim = Adam(net,lr=lr)

net.train_on(optimizer=optim,
        lossfn=lossfn,
        trainloader=train_loader,
        testloader = test_loader,
        save_weights_in='FC',
        epochs = epochs,
)
import nnpy.nn as nn

net = nn.Sequential([
    nn.Linear(28*28,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10),
    nn.Softmax(),
])
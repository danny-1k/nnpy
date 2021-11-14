import os
import numpy as np
import nnpy.nn as nn


class FC(nn.Module):
    def __init__(self):

        self.fc1 = nn.Linear(28*28,128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128,64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64,10)
        self.soft = nn.Softmax()

        self.drop = nn.Dropout(.3)


    def forward(self,x):
        x = self.relu1(self.fc1(x))
        
        x = self.drop(x)

        x = self.relu2(self.fc2(x))

        x = self.soft(self.fc3(x))

        return x

    
    def save_weights(self):
        if 'fc_model' not in os.listdir('weights'):
            print('Weights folder does not exist. Creating')
            os.makedirs('weights/fc_model')

        
        for layer_idx,layer in enumerate(self.layers):
            for param in layer.params:
                np.save(f'weights/fc_model/layeridx_{layer_idx}_param_{param}',layer.params[param])
    

    def load_weights(self):
        if 'fc_model' not in os.listdir('weights'):
            print('Weights folder does not exist. Creating')
            os.makedirs('weights/fc_model')

        for layer_idx,layer in enumerate(self.layers):
            for param in layer.param:
                param_file = f'layeridx_{layer_idx}_param_{param}'
                if param_file not in os.listdir('weights/fc_model'):
                    print(f'Param for {param} in layer {layer}({layer_idx}) not found')
                self.layers[layer_idx][param] = np.load(f'weights/fc_model/{param_file}')
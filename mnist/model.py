import os
import numpy as np
import nnpy.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt

class Base:

    def save_weights(self,save_weights_in):
        if save_weights_in not in os.listdir('weights'):
            print('Weights folder does not exist. Creating')
            os.makedirs(f'weights/{save_weights_in}')

        
        for layer_idx,layer in enumerate(self.layers):
            for param in layer.params:
                np.save(f'weights/{save_weights_in}/layeridx_{layer_idx}_param_{param}',layer.params[param])
    
    def load_weights(self,save_weights_in):
        if save_weights_in not in os.listdir('weights'):
            print('Weights folder does not exist. Creating')
            os.makedirs(f'weights/{save_weights_in}')

        for layer_idx,layer in enumerate(self.layers):
            for param in layer.param:
                param_file = f'layeridx_{layer_idx}_param_{param}'
                if param_file not in os.listdir(f'weights/{save_weights_in}'):
                    print(f'Param for {param} in layer {layer}({layer_idx}) not found')
                self.layers[layer_idx][param] = np.load(f'weights/{save_weights_in}/{param_file}')

    
    def train_on(self,optimizer,lossfn,trainloader,testloader,save_weights_in,plots_folder='plots',epochs=10,):
        
        test_loss_over_time = []
        train_loss_over_time = []
        acc_over_time = [0]


        for epoch in tqdm(range(epochs)):
            train_batch_loss = []
            test_batch_loss = []
            accuracy = 0

            self.train()

            for x,y in trainloader:
                x = x.view(-1,28*28).numpy()
                y = y.numpy()

                p = self.__call__(x)

                loss = lossfn(p,y)

                train_batch_loss.append(loss)

                lossfn.backward()

                optimizer.step()
                optimizer.zero_grad()


            self.eval()

            for x,y in testloader:
                x = x.view(-1,28*28).numpy()
                y = y.numpy()

                p = self.__call__(x)

                loss = lossfn(p,y)

                test_batch_loss.append(loss)

                accuracy += sum(p.argmax(axis=1) == y)
            
            accuracy = accuracy/len(testloader.dataset)
            acc_over_time.append(accuracy)

            train_loss_over_time.append(np.mean(train_batch_loss))
            test_loss_over_time.append(np.mean(test_batch_loss))

            if acc_over_time[-1] > acc_over_time[-2]:
                self.save_weights(save_weights_in)


            plt.plot(train_loss_over_time,legend='train')
            plt.plot(test_loss_over_time,legend='test')

            plt.legend()

            plt.savefig(os.path.join(self.plots_folder,'loss.png'))
            
            plt.close('all')


            plt.plot(acc_over_time,legend='accuracy')

            plt.legend()

            plt.savefig(os.path.join(plots_folder,'accuracy.png'))
            
            plt.close('all')



class FC(nn.Module,Base):
    def __init__(self):

        self.fc1 = nn.Linear(28*28,128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128,64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64,10)
        self.soft = nn.Softmax()

        self.drop = nn.Dropout(.3)


    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        
        x = self.drop(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.soft(x)

        return x
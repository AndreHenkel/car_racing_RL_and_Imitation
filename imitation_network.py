import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

import utils

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LEARNING_RATE = 1e-4

class Imitation_Network(nn.Module):

    def __init__(self,input_shape, action_size,seed = 1337):

        super(Imitation_Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = input_shape
        self.action_size = action_size

        self.act_func = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4).cuda()
        self.conv2 = nn.Conv2d(32, 64, 4,stride=2).cuda()
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1).cuda()

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(4096,512).cuda()
        self.fc2 = nn.Linear(512,action_size).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        #self.optimizer = optim.Adam(self.parameters(),lr = 1e-3)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=LEARNING_RATE, eps=1e-2, alpha=0.95)
        self.loss_func = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        #nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        #print("Imitation_Network.reset_parameters -> Not implemented yet")


    def forward(self, state):
        x = torch.reshape(state,(state.shape[0],1,96,96))
        x = self.conv1(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x= self.conv3(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        #print(x.shape)
        x = self.act_func(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x) #to trim actions into range -1 to +1
        #x = self.softmax(x)
        return x


    def update_model(self,states,actions):
        """ taking in batch of states and actions. Have to be the same length """
        self.optimizer.zero_grad()
        predictions = self.forward(states)
        #print(predictions)
        #print(states[0],actions[0],predictions[0])
        target = utils.action_id2arr(torch.argmax(actions,dim=1).cpu())
        #print(target)
        mse_loss = F.mse_loss(predictions, torch.from_numpy(target).cuda())
        #cel = self.loss_func(predictions.float(),torch.argmax(actions,dim=1))
        #print("cel: ", cel)
        #cel.backward()
        mse_loss.backward()
        self.optimizer.step()

        return mse_loss.item()

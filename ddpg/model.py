#DDPG model, including actor and critic
#parameters from the paper: https://arxiv.org/pdf/1509.02971.pdf
# used for pixel input

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torchvision import transforms
import numpy as np
import math


##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UNIFORM_INIT_ACTOR = 3e-4
UNIFORM_INIT_CRITIC = 3e-4


resnet18 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).to(device)
resnet18.eval()
preprocess = transforms.Compose([
    transforms.Lambda(lambda x: x/255), #put image in range of 0,1
    transforms.Resize(224),
    transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def init_uniform_rule(m):
    if type(m) == nn.Linear:
        #linear init as described in paper
        n = m.in_features
        y = 1.0/np.sqrt(n)
        #init.xavier_uniform_(m.weight, gain=0.1)
        m.weight.data.uniform_(-y,y)
        m.bias.data.uniform_(-y,y)
        #print("Init linear")

    if type(m) == nn.Conv2d:
        #conv2d standard init
        init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)
        #print("Init Conv2d")
    return True

class Actor(nn.Module):
    def __init__(self,input_shape, action_space,seed = 1337):
        """
        action_space being a gym.Box object.
        """
        super(Actor,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = input_shape
        self.action_size = 3

        self.act_func = nn.ReLU()

        self.fc1 = nn.Linear(1000,400).cuda()
        self.fc2 = nn.Linear(400,300).cuda()
        self.output = nn.Linear(300,3).cuda()
        self.tanh = nn.Tanh()

        #self.optimizer = optim.Adam(self.parameters(),lr = 1e-3)

        self.reset_parameters()


    def reset_parameters(self):
        """
        Init last layers from a uniform distribution [-3×10−4,3×10−4]
        """
        self.apply(init_uniform_rule)
        # self.fc1.weight.data.uniform_(-0.1, 0.1)
        # self.fc2.weight.data.uniform_(-UNIFORM_INIT_ACTOR,UNIFORM_INIT_ACTOR)
        # self.fc2.bias.data.uniform_(-UNIFORM_INIT_ACTOR,UNIFORM_INIT_ACTOR)
        self.output.weight.data.uniform_(-UNIFORM_INIT_ACTOR,UNIFORM_INIT_ACTOR)
        #self.output.weight.data.xavier_uniform_()
        #self.output.bias.data.xavier_uniform_()

    #@profile
    def forward(self, input):
        x = torch.reshape(input,(input.shape[0],3,96,96))
        with torch.no_grad():
            x = preprocess(x)
            x = resnet18(x)

        #print(x.shape)
        x = self.act_func(self.fc1(x))
        x = self.act_func(self.fc2(x))
        x = self.tanh(self.output(x)) #to trim actions into range -1 to +1
        #x = self.softmax(x)
        return x


class Critic(nn.Module):
    def __init__(self,input_shape, seed = 1337):
        """
        input_shape of the image including channels
        output_size for critic is 1, since it is just a Q-value
        """
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = input_shape
        #self.action_size = action_size

        self.act_func = nn.ReLU()
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(1000,400).cuda()
        self.fc2 = nn.Linear(403,300).cuda()
        self.output = nn.Linear(300,1).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        """
        Init from a uniform distribution [-3×10−4,3×10−4]
        """
        self.apply(init_uniform_rule)
        # self.fc1.weight.data.uniform_(-0.1, 0.1)
        # self.fc2.weight.data.uniform_(-0.1, 0.1)
        # self.fc2.weight.data.uniform_(-UNIFORM_INIT_CRITIC,UNIFORM_INIT_CRITIC)
        # self.fc2.bias.data.uniform_(-UNIFORM_INIT_CRITIC,UNIFORM_INIT_CRITIC)
        self.output.weight.data.uniform_(-UNIFORM_INIT_CRITIC,UNIFORM_INIT_CRITIC)
        #self.output.weight.data.xavier_uniform_()
        #self.output.bias.data.xavier_uniform_()

    #@profile
    def forward(self, input, action):
        """
        As described in the paper, include action into the 2nd FC layer, or the first one. Not definitely clear by reading the paper
        """
        x = torch.reshape(input,(input.shape[0],3,96,96))
        with torch.no_grad():
            x = preprocess(x)
            x = resnet18(x)
        x = self.act_func(self.fc1(x))
        x = torch.cat((x,action), dim=1)
        x = self.act_func(self.fc2(x))
        x = self.output(x)
        return x

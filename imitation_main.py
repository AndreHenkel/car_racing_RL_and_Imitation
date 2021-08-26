import gym
import utils
import os

import numpy as np
import torch
from imitation_network import Imitation_Network

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#env = gym.make("CarRacing-v0")
#obs = env.reset()

states = 0
actions = 0
init = False
#read data from datasets folder
directory = r'./data/'
for n,filename in enumerate(os.listdir(directory)):
    if filename.endswith(".pkl.gzip"):
        sg_file_path = os.path.join(directory, filename)
        if not init:
            states,actions,r,n_s,d = utils.read_data(sg_file_path)
            init = True
        else:
            X,Y = utils.read_data(sg_file_path)
            states = np.concatenate((states,X))
            actions = np.concatenate((actions,Y))
    else:
        continue

    if n >=20: #to stop early for debugging
        break

print(states.shape)
#preprocess states
states_pp = utils.preprocess_state(states)
actions_pp = utils.transl_action_env2agent(actions)
print("Preprocessed")
#balance actions out
X,Y = utils.balance_actions(states_pp,actions_pp,0.8)# | 0.8 worked so far kind of well
print("Balanced out")
print(X.shape)
#train
net = Imitation_Network((96,96),3) #for continuous output

batch_size = 32 # | 32 worked good
epochs = 500000 # | 200000 worked good
cel_hist = []
for i in range(epochs):
    indices = np.random.randint(0,X.shape[0],batch_size)
    cel = net.update_model(torch.from_numpy(X[indices]).to(device),torch.from_numpy(Y[indices]).to(device))
    #print(Y[indices[0]])
    #utils.show_img(X[indices[0]])

    cel_hist.append(cel)

    if i%1000 == 0 and i != 0:
        print("Epoch number: ",i)
        print(np.mean(cel_hist[i-1000:i]))

    #store in between to test
    if i%50000 == 0 and i != 0:
        torch.save(net.state_dict(),'net.torch_%d' %i)

#torch.save(net.state_dict(),'net.torch')

#utils.show_img(states_pp[0])
#utils.show_run(states_pp[0:50])
#print(states_pp[0:2].shape)


#let it run in env and see what happens
# env = gym.make("CarRacing-v0")
# obs = env.reset()
# done = False
# rewards = []
# while done == False:
#     obs_pp = utils.preprocess_state(obs)
#     prediction = net(obs_pp)
#     action = utils.action_id2arr(torch.argmax(prediction))
#     obs,rew,done,info = env.step(action)
#     rewards.append(rew)
#     env.render()
#
# print(sum(rewards))

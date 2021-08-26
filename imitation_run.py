import gym
import utils
import os
import time
import numpy as np
import torch
from imitation_network import Imitation_Network

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Imitation_Network((96,96),3)


artifical_speed_up = 0 #40
torch_model = 'net.torch_10000'
do_render = True



net.load_state_dict(torch.load(torch_model))
net.eval()

#utils.show_img(states_pp[0])
#utils.show_run(states_pp[0:50])
#print(states_pp[0:2].shape)


#let it run in env and see what happens
env = gym.make("CarRacing-v0")

#video_recorder = VideoRecorder(
#        env, "Video.mp4",enabled=true)

#env = gym.wrappers.Monitor(env, 'recording')

s_t = 0
print("Sleep for %s seconds",s_t)
time.sleep(s_t)

go = [0.0,1.0,0.0]


ep_rewards = []

for i in range(40):
    rewards = []
    states = []
    done = False
    obs = env.reset()
    for i in range(artifical_speed_up):
        env.step(go)

    while done == False:
        states.append(obs)
        obs_pp = utils.preprocess_state(np.reshape(obs,(1,96,96,3)).astype(np.float))
        action = net.forward(torch.from_numpy(obs_pp).float().to(device))
        action = action.cpu().detach().numpy()
        action = action[0]
        #action = utils.action_id2arr(torch.argmax(action))

        print(action)
        obs,rew,done,info = env.step(action)
        rewards.append(rew)
        if do_render:
            env.render()

        #if len(states)==50:
        #    utils.show_img(obs,c='color')
        #    utils.show_img(obs_pp,c='gray')

    ep_rewards.append(sum(rewards))
    print(i)
    print(sum(rewards))

#video_recorder.close()
#utils.store_video(states)
ep_rewards = np.asarray(ep_rewards)
print(ep_rewards)
print(np.mean(ep_rewards))

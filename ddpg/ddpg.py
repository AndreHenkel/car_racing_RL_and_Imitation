#This is the control class, that implements the ddpg algorithm

#import standard classes
import gym
import copy
import numpy as np
import cv2
import sys
from guppy import hpy
import torch
import random
import os
import matplotlib.pyplot as plt


# import own classes
from ddpg.agent import Agent
from ddpg.ou_noise import OUNoise as ou
import utils


#@profile
def ddpg(n_episodes=500,max_steps=1000,base_models=(None,None),imitation_actor=None, memory_size=int(1e6),use_training_data=False):
    """
    n_episodes: number of episodes to run through
    max_steps: maximum time spent in one episode (steps)
    base_models can be used to start from already trained models (actor,critic) tuple

    -ddpg is an off-policy algorithm.
    -update model with minibatch on each step(possibly each x steps)

    """
    if use_training_data:
        # NOTE: uses only last file in this directory
        im_states, im_actions, im_rewards, im_next_states, im_dones = (0,0,0,0,0)
        directory = r'./data/'
        for n,filename in enumerate(os.listdir(directory)):
            if filename.endswith(".pkl.gzip"):
                sg_file_path = os.path.join(directory, filename)
                im_states,im_actions,im_rewards,im_next_states,im_dones = utils.read_data(sg_file_path)
                im_states[:, 85:, :15, :] = [0.0, 0.0, 0.0]
                im_next_states[:, 85:, :15, :] = [0.0, 0.0, 0.0]
                print("Blacked out reward in states and next_states for imitation data")
                print("Loaded imitation data")
            else:
                continue
        step_nr = 0
        print("states.shape(): "        ,im_states.shape)
        print("actions.shape(): "       ,im_actions.shape)
        print("rewards.shape(): "       ,im_rewards.shape)
        print("next_states.shape(): "   ,im_next_states.shape)
        print("dones.shape(): "         ,im_dones.shape)

        #normalize data
        # im_states = im_states / 255
        # im_next_states = im_next_states / 255
        #print("Normalized states in pretrain data")


    #h = hpy()
    #init environment
    env = gym.make("CarRacing-v0")
    observation = env.reset() # (96,96,3)
    #print(observation.shape)
    #observation = np.reshape(observation,(1,96,96,3))
    #observation = utils.preprocess_state(np.reshape(observation,(1,observation.shape[0],observation.shape[1],observation.shape[2])).astype(np.float))
    #using rgb with score etc.
    #observation = utils.preprocess_state(np.reshape(observation,(1,96,96,3)).astype(np.float))

    #resizing to 64x64 as in the paper
    #observation = np.asarray([cv2.resize(observation[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])
    # 4 state MODE
    #obs = observation
    #observation = utils.conc_states([obs,obs,obs,obs])[0] #same amount as state_gathering of obs
    input_shape = observation.shape
    action_space = env.action_space

    print("input_shape: ",input_shape)
    print("action_space: ", action_space)

    agent = Agent(input_shape, action_space, state_gathering=4, memory_size=memory_size)


    if base_models[0] != None and base_models[1] != None:
        print("Loading state_dicts for training")
        agent.load(base_models[0], base_models[1], base_models[2],
                    base_models[3], base_models[4], base_models[5])

    if imitation_actor != None:
        agent.load_imitation_actor(imitation_actor)

    def flush_imit(steps = 20000):
        if use_training_data:
            if steps>len(im_states):
                steps = len(im_states)
            idx = np.random.randint(0,X.shape[0],steps) #random sample
            step = 0
            for s, a, r, n_s, d in zip(im_states[idx], im_actions[idx], im_rewards[idx], im_next_states[idx], im_dones[idx]):
                #s = utils.preprocess_state(np.reshape(s,(1,96,96,3)).astype(np.float))
                s = np.reshape(s,(1,96,96,3)).copy()
                #n_s = utils.preprocess_state(np.reshape(n_s,(1,96,96,3)).astype(np.float))
                n_s = np.reshape(n_s,(1,96,96,3)).copy()
                #print(a[1])
                #if np.mean(a) != 0.0 or np.random.randn() > 0.5: #remove half of the "do-nothing"-actions
                agent.step(s,a,r,n_s,d)
                step = step + 1
                if step%1000 == 0 and step > 1000:
                    print(np.mean(np.asarray(agent.critic_loss_over_time[len(agent.critic_loss_over_time)-1000:])))
                    #plt.plot(agent.critic_loss_over_time)

                if step >= steps:
                    break
            print("Trained agent with imitation data for: ", step)

    #train on imitation data for x steps and save the models after each step
    if use_training_data:
        for i in range(1):
            flush_imit()
            agent.save(i)

        # loss_list = []
        # for i in range(int(len(agent.critic_loss_over_time)/100)):
        #     loss_list.append(np.mean(np.asarray(agent.critic_loss_over_time[i*100:(i+1)*100-1])))
        # plt.plot(loss_list)
        # plt.show()


    noise_decay = 0.95
    scores_per_episode = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state[85:, :15, :] = [0.0, 0.0, 0.0] #black out current rewards from state pixel
        #state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
        state = np.reshape(state,(1,96,96,3))
        #state = state / 255
        # state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
        # state = np.asarray([cv2.resize(state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])

        noise_decay = noise_decay ** i_episode
        #3 action size, randn is mu
        #noise = ou(3)

        episode_score = 0
        for s in range(max_steps):
            # action = torch.from_numpy(state).float().to(device).
            # action = action.cpu().detach().numpy()
            action = agent.act(state)[0]
            #print("action: ",action[0])
            if not use_training_data: #only add if not already trained with training data beforehand
                action = action + np.random.normal(loc=0.0, scale=0.1,size=3)*noise_decay #noise normal
            #action = action[0]+ noise.sample() #[0] getting values from tensor
            action = np.clip(action, -1, 1)
            # if random.random() <= 0.1:
            #     action = [0.0, 0.7, 0.0] #just go straight with acceleration

            #print("Resulting action: ", action)
            next_state,reward,done,info = env.step(action)
            next_state[85:, :15, :] = [0.0, 0.0, 0.0] #black out current rewards from next_state pixel
            #next_state = utils.preprocess_state(np.reshape(next_state,(1,96,96,3)).astype(np.float))
            next_state = np.reshape(next_state,(1,96,96,3))
            # next_state = next_state / 255
            # next_state = utils.preprocess_state(np.reshape(next_state,(1,96,96,3)).astype(np.float))
            # next_state = np.asarray([cv2.resize(next_state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)]) #resizing to 64x64 as in the paper

            agent.step(state,action,reward,next_state,done)

            state = next_state
            episode_score += reward

            if done:
                break

        scores_per_episode.append(episode_score)

        #create new one, so that the memory gets cleaned
        env.close()
        env = gym.make("CarRacing-v0")


        if use_training_data:
            flush_imit(1000)
        print("Episode number: ",i_episode)
        print("Episode score: ", episode_score)
        print("Memory size: ", agent.get_memory_size())
        print("Critic loss: ", np.mean(np.asarray(agent.critic_loss_over_time[-100:])))

        if i_episode%2==0 and i_episode > 0:
            print("Storing ddpg.torch_xxx...")
            agent.save(i_episode)

import gym
import cv2
import utils
import numpy as np
import sys
import time
print(sys.argv)


# a = np.array([[[1,2,3],[4,5,6],[7,8,9]],
#             [[11,22,33],[44,55,66],[77,88,99]],
#             [[111,222,333],[444,555,666],[777,888,999]]])
#
# print(a.shape)
#
# a =  np.reshape(state,(1,state.shape[2],state.shape[0],state.shape[1]))
# print(a.shape)


#time.sleep(10)

env = gym.make("CarRacing-v0")
state = env.reset()
for i in range(50):
    env.step([0.0,0.0,0.0])
print(state.shape)
state = np.reshape(state,(1,96,96,3))
utils.show_img(state[0], "color")

states = []
state = env.reset()
for _ in range(10):
    s,r,d,i = env.step([0.0,1.0,0.0])
    states.append(s)

cs = np.concatenate(np.asarray(states), axis=2)

print(cs.shape)

state,action,reward,done = env.step([0.0,0.5,0.0])
print(state.shape)
#state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
#scaled = cv2.resize(state[0], dsize=(64,64), interpolation=cv2.INTER_NEAREST)


a = env.action_space
print(a.shape[0])
print(a.sample())

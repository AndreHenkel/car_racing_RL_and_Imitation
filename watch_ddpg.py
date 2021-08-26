
from ddpg.agent import Agent

import utils
import gym
import numpy as np
import torch
import cv2
from pyglet.window import key
import sys




##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

STEERING = 1.0
GAS = 5.0
BRAKE = 0.4

a = np.zeros(3, dtype=np.float32)
# (steer,gas,brake)

op_mode = False

def key_press(k, mod):
    global restart
    global end_game
    global op_mode
    if k == 0xff0d: restart = True
    if k == key.ESCAPE: end_game = True
    if k == key.UP or k == key.W:
        a[1] = +GAS
        a[2] = 0.0 #remove brake
        #if a[0] = 0.0:
        #     a[1] = +GAS
    if k == key.LEFT or k == key.A:
        a[0] = -STEERING
        # a[1] =  0.0  # Cut gas while turning
    if k == key.RIGHT or k == key.D:
        a[0] = +STEERING
        # a[1] =  0.0  # Cut gas while turning
    if k == key.DOWN or k == key.S:
        a[2] = +BRAKE  # stronger brakes
    if k == key.SPACE:
        print("test")
        if op_mode:
            op_mode = False
        else:
            op_mode = True

def key_release(k, mod):
    if (k == key.LEFT or k == key.A) and a[0] == -STEERING:
        a[0] = 0.0
        # if a[3] == STEERING:
        #     a[1] = STEERING
    if (k == key.RIGHT or k == key.D) and a[0] == +STEERING:
        a[0] = 0.0
        # if a[3] == STEERING:
        #     a[1] = STEERING
    if k == key.UP or k == key.W:
        a[1] = 0.0
        # a[3] = 0.0
    if k == key.DOWN or k == key.S:
        a[2] = 0.0


env = gym.make('CarRacing-v0').unwrapped
env.reset()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
observation = env.reset()
observation = np.reshape(observation,(1,96,96,3))
#observation = utils.preprocess_state(np.reshape(observation,(1,96,96,3)).astype(np.float))
# observation = np.reshape(observation,(1,96,96,3))
# observation = observation / 255
# observation = utils.preprocess_state(np.reshape(observation,(1,observation.shape[0],observation.shape[1],observation.shape[2])).astype(np.float))
# observation = np.asarray([cv2.resize(observation[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])
#4 state MODE
#observation = utils.conc_states([obs,obs,obs,obs])[0] #same amount as state_gathering of obs
input_shape = observation.shape
action_space = env.action_space

print("input_shape: ",input_shape)
print("action_space: ", action_space)

agent = Agent(input_shape,action_space)

model_nr = 100#default
if len(sys.argv) == 2:
    print("Using model number: ",sys.argv[1])
    model_nr = int(sys.argv[1])

agent_model = torch.load("ddpg_models/actor.torch_"+str(model_nr))
critic_model = torch.load("ddpg_models/critic.torch_"+str(model_nr))
agent_target_model = torch.load("ddpg_models/actor_target.torch_"+str(model_nr))
critic_target_model = torch.load("ddpg_models/critic_target.torch_"+str(model_nr))
actor_optimizer_model = torch.load("ddpg_models/actor_optimizer.torch_"+str(model_nr))
critic_optimizer_model = torch.load("ddpg_models/critic_optimizer.torch_"+str(model_nr))
agent.load(agent_model, critic_model,agent_target_model,critic_target_model,actor_optimizer_model,critic_optimizer_model)


for i in range(40):
    state = env.reset()
    # state = np.reshape(state,(1,96,96,3))
    # state = state/255
    #state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
    state = np.reshape(state,(1,96,96,3)).copy()
    # state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
    # state = np.asarray([cv2.resize(state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])

    for s_i in range(1000):
        action = agent.act(state)
        #action = agent.target_pred(state)
        if op_mode:
            action[0] = a
        action[0][2] = 0.0
        if action[0][1] < 0:
            action[0][1] *= -1
        print(action[0])
        next_state,reward,done,info = env.step(action[0])
        #next_state = utils.preprocess_state(np.reshape(next_state,(1,96,96,3)).astype(np.float))
        next_state = np.reshape(next_state,(1,96,96,3)).copy()
        # next_state = np.reshape(next_state,(1,96,96,3))
        # next_state = next_state / 255
        # next_state = utils.preprocess_state(np.reshape(next_state,(1,96,96,3)).astype(np.float))
        # next_state = np.asarray([cv2.resize(state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])

        state = next_state
        env.render()

        if done:
            break

#######################
#4 gathering MODE
#######################
# for i in range(40):
#     state = env.reset()
#
#     artifical_speed_up = 50
#     go = [0.0, 1.0,0.0]
#     for i in range(artifical_speed_up):
#         state,r,d,_ = env.step(go)
#
#     state = utils.preprocess_state(np.reshape(state,(1,96,96,3)).astype(np.float))
#     state = np.asarray([cv2.resize(state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])
#     done = False
#     gathered_states = [state,state,state,state]
#
#     for s_i in range(1000):
#         #rotate
#         gathered_states[0] = gathered_states[1]
#         gathered_states[1] = gathered_states[2]
#         gathered_states[2] = gathered_states[3]
#         gathered_states[3] = state
#         c_state = utils.conc_states(gathered_states)
#         c_state = torch.from_numpy(c_state).float().to(device)
#         agent.actor_local.eval()
#         with torch.no_grad():
#             action = agent.actor_local(c_state).cpu().data.numpy()
#         agent.actor_local.train()
#         action=  np.clip(action, -1, 1)
#
#         #if s_i % 50==0 and s_i > 0:
#         #    utils.show_img(gathered_states[3])
#
#         next_state,reward,done,info = env.step(action[0])
#         next_state = utils.preprocess_state(np.reshape(next_state,(1,96,96,3)).astype(np.float))
#         next_state = np.asarray([cv2.resize(next_state[0], dsize=(64,64), interpolation=cv2.INTER_AREA)])
#         agent.step(state,action,reward,next_state,done)
#         if done:
#             break
#         state = next_state
#         env.render()

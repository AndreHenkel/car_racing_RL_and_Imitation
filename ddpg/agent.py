# Code partly taken and adapted from: https://github.com/AndreHenkel/p2_continuous_control
# This code follows this paper: https://arxiv.org/abs/1509.02971

# This class incorporates the agent that interacts with the environment

#####################################
# target models                     #
#####################################
# these copies of the Actor and Critic are used to calculate the target values
# the weights of the normal networks are then "soft" copied to the target network
# that allows for smoother training.
# without that the agent might be prone to do what he just learned,
# instead of waiting a bit about what is to come in the future
#####################################

import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from utils import conc_states


# import own classes
from ddpg.model import Actor,Critic
from ddpg.replay_buffer import ReplayBuffer

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16 #paper 16
GAMMA = 0.99 #paper 0.99
LEARNING_RATE_ACTOR = 1e-4  #from paper for actor 1e-4
LEARNING_RATE_CRITIC = 1e-3  #from paper for critic 1e-3
WEIGHT_DECAY = 0.00001  # L2 weight decay
TRAIN_X_STEP = 20
TRAIN_X_STEP_FOR = 10
TAU = 1e-3 #soft update, paper: 1e-3

class Agent:
    def __init__(self, input_shape, action_space, state_gathering=4, seed=9037, memory_size=int(1e6)):
        """
        @args:
            -input_shape: Expected to be an image array with channels of form (Width, Height, Channels)
            -action_space: Expected to be of type gym.Box
        """
        self.input_shape = input_shape
        self.action_space = action_space
        self.learn_cnt = 0 # counter for learning every x-th step
        #initialize classes
        self.actor_local = Actor(input_shape,action_space, seed=seed)
        self.actor_target = Actor(input_shape,action_space, seed=seed)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)
        self.actor_optimizer = torch.optim.RMSprop(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR, eps=1e-2, alpha=0.95)
        #self.actor_optimizer = optim.RMSprop(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)

        self.critic_local = Critic(input_shape, seed=seed)
        self.critic_target = Critic(input_shape, seed=seed)
        self.critic_optimizer =  optim.Adam(self.critic_local.parameters(),lr = LEARNING_RATE_CRITIC, weight_decay=WEIGHT_DECAY) #from paper
        #self.critic_optimizer = torch.optim.RMSprop(self.critic_local.parameters(), lr=LEARNING_RATE_CRITIC, eps=1e-2, alpha=0.95,  weight_decay=WEIGHT_DECAY)
        #self.critic_optimizer =  optim.RMSprop(self.critic_local.parameters(),lr = LEARNING_RATE_CRITIC, weight_decay=WEIGHT_DECAY) #from paper

        self.memory = ReplayBuffer(memory_size) #using default max_exp
        self.critic_loss_over_time = []

        # 4 state mode
        # self.state_gathering = state_gathering
        # self.gathered_action = [0.0, 0.0, 0.0]
        # self.gathered_rewards = []
        # self.gathered_states = []
        # self.gathered_next_states = []

        print("Agent initialized")

    def load(self, actor_dict, critic_dict, actor_target_dict,
            critic_target_dict, actor_optimizer_dict=None, critic_optimizer_dict=None):
        self.actor_local.load_state_dict(actor_dict)
        self.critic_local.load_state_dict(critic_dict)
        self.actor_target.load_state_dict(actor_target_dict)
        self.critic_target.load_state_dict(critic_target_dict)
        #optimizer
        if actor_optimizer_dict!=None:
            self.actor_optimizer.load_state_dict(actor_optimizer_dict)
        if critic_optimizer_dict!=None:
            self.critic_optimizer.load_state_dict(critic_optimizer_dict)

    def load_imitation_actor(self, imitation_actor):
        self.actor_local.load_state_dict(imitation_actor)
        self.actor_target.load_state_dict(imitation_actor)
        print("Loaded imitation_actor")


    def save(self, number):
        torch.save(self.actor_local.state_dict(),'ddpg_models/actor.torch_%d' %number)
        torch.save(self.critic_local.state_dict(),'ddpg_models/critic.torch_%d' %number)
        torch.save(self.actor_target.state_dict(),'ddpg_models/actor_target.torch_%d' %number)
        torch.save(self.critic_target.state_dict(),'ddpg_models/critic_target.torch_%d' %number)
        #save optimizer
        torch.save(self.actor_optimizer.state_dict(),'ddpg_models/actor_optimizer.torch_%d' %number)
        torch.save(self.critic_optimizer.state_dict(),'ddpg_models/critic_optimizer.torch_%d' %number)

        print("Saved agent state_dicts with: ",number)


    def get_memory_size(self):
        return len(self.memory)

    #@profile
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # 4 state MODE
        # #gather n-steps to concatenate states to update from
        # if len(self.gathered_states) >=self.state_gathering:
        #     self.gathered_next_states.append(next_state)
        #     self.gathered_rewards.append(reward)
        #     if len(self.gathered_next_states) >= self.state_gathering:
        #         #concatenate states and next_states
        #         c_state = conc_states(self.gathered_states)
        #         c_next_state = conc_states(self.gathered_next_states)
        #
        #         # print("c_state.shape: ", c_state.shape)
        #         # print("c_next_state.shape: ", c_next_state.shape)
        #         self.memory.add(c_state,self.gathered_action,sum(self.gathered_rewards),c_next_state,done)
        #         self.gathered_states = self.gathered_next_states
        #         self.gathered_next_states = []
        #         #taking action and reward from the transition phase from c_state to c_next_state
        #         self.gathered_action = action
        # else:
        #     self.gathered_states.append(state)
        #     self.gathered_action = action
        #     self.gathered_rewards.append(reward)
        #
        # if done:
        #     #reset
        #     self.gathered_states = []
        #     self.gathered_next_states = []

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            # using the 10/20 "trick"
            # only train every 10/20 steps, but then train for X rounds
            if self.learn_cnt >= TRAIN_X_STEP:
                for _ in range(TRAIN_X_STEP_FOR):
                    exps = self.memory.sample(BATCH_SIZE)
                    cl = self.learn(exps, GAMMA)
                    self.critic_loss_over_time.append(cl)
                self.learn_cnt = 0 #reset counter
            else:
                self.learn_cnt+=1

    def target_pred(self,state):
        state = torch.from_numpy(state).float().to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_target.train()
        return action #np.clip(action, -1, 1)

    def act(self, state):
        """Returns actions for given state as per current policy.
           Deactivates training for gathering the current actions
           and activates training mode afterwards again
        """
        #4 state MODE
        #if len(self.gathered_states) >= self.state_gathering:
        #state = conc_states(self.gathered_states)
        state = torch.from_numpy(state.copy()).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return np.clip(action, -1, 1)
        # 4 state MODE
        # else:
        #     return [self.action_space.sample()]

    #@profile
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 0.0000001)
        rewards = torch.clip(rewards, -1, 1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        #TODO: Implement ddpg-argmax here.
        # samples 100 actions for each next_state and takes the argmax of it.
        actions_next_argmax = torch.empty((states.shape[0],3),device='cuda:0')

        action_samples = 100

        for i,n_s in enumerate(next_states):
            n_s = torch.reshape(n_s,(1,3,96,96))
            #actions_next = self.actor_target(n_s.cuda())
            q_tmp = -1000
            a_n_argmax = 0
            for _ in range(action_samples):
                action = torch.from_numpy(self.action_space.sample()).float().cuda()
                action = torch.reshape(action,(1,3))
                #print(action[0])
                Q_targets_next = self.critic_target(n_s, action)
                #print(Q_targets_next[0])
                if Q_targets_next[0] > q_tmp:
                    q_tmp = Q_targets_next[0]
                    a_n_argmax = action[0]

            actions_next_argmax[i] = a_n_argmax

        #actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next_argmax)



        #print("Q: ", Q_targets_next)
        # Compute Q targets for current states (y_i)

        self.critic_optimizer.zero_grad()
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        #print("Q_expected: ", Q_expected) #increasing super fast
        #print("Q_targets: ", Q_targets)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        #print("Critic_loss: ", critic_loss)

        # Minimize the loss
        #self.critic_optimizer.zero_grad()
        #counter exploding gradients problem
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        self.actor_optimizer.zero_grad()
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        #self.actor_optimizer.zero_grad()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),0.5)
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(),1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_local, self.actor_target, TAU)
        return critic_loss.item()


    #@profile
    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def _random_action(self):
        return self.action_space.sample()

    def _batch_normalization(self,batch):
        print("Implement _batch_normalization")
        return batch

# This module is used to store and retrieve experience gathered during the ddpg training

from collections import deque, namedtuple
import numpy as np
import random
import torch

##decide weather to use gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EXP = namedtuple('EXP', ['state','action','reward','next_state','done'])
EXPS = namedtuple('EXPS', ['states','actions','rewards','next_states','dones'])

class ReplayBuffer:
    def __init__(self, max_exp=int(1e4)):
        """
        Default max_exp is 1e6, as described in the paper
        """
        self.max_exp = max_exp
        self.data = deque(maxlen=max_exp)

    #@profile
    def add(self, state, action, reward, next_state, done):
        exp = EXP(state,action,reward,next_state,done)
        self.data.append(exp)

    #@profile
    def sample(self, sample_size=16):
        """
        sample_size: amount of experience being returned. Default 16 as described in the paper for pixel input
        @return: namedtuple including arrays of states,actions,rewards,next_states,dones
        """
        if sample_size <= len(self):
            experiences = random.sample(self.data,sample_size)

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                device)

            exps = EXPS(states,actions,rewards,next_states,dones)
            return exps
        else:
            raise Exception("ReplayBuffer.sample(self,sample_size) -> sample_size is larger than the ReplayBuffer")

    def _normalize_batch(self, batch_sample):
        print("Implement Replaybuffer._normalize_batch()")

    def __len__(self):
        return len(self.data)

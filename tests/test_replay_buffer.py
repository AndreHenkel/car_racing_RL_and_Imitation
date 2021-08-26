from ddpg.replay_buffer import ReplayBuffer

import numpy as np

rb = ReplayBuffer(2)
T = True

for i in range(int(1e100)):
    s = np.random.randint((70,70,70))
    rb.add(s,9,9,s,T)


print(len(rb))
exps = rb.sample(2)
print(exps)
print(exps.states)

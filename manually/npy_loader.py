import numpy as np

f = open("datasets/human_expert_11.15.2020_19:43:08.npy",'rb')

for i in range(10):
    s = np.load(f)
    a=np.load(f)
    r=np.load(f)
    print(s,a,r)

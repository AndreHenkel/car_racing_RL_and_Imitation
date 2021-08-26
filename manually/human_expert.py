import numpy as np
import gym
from datetime import datetime

from gym.utils import EzPickle

#human driving code taken from and slightly adapted to use human driving as data pool for training:
# https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py


#config
record_game = True

if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = gym.make("CarRacing-v0")
    env.render()
    print("max_episode_steps: ",env.spec.max_episode_steps)

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True

    now = datetime.now() # current date and time
    f = open("../datasets/human_expert_"+now.strftime("%m.%d.%Y_%H:%M:%S")+".npy","wb");

    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if record_game:
                np.save(f,s)
                np.save(f,a)
                np.save(f,r)
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                print(done,restart,isopen)
                break
    env.close()
    f.close()

import gym

env = gym.make("CarRacing-v0")
obs = env.reset()


action = (0,1,0)
while True:
    obs, rew, done, info = env.step(action)
    env.render("human")

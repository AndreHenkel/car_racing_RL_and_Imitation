from ddpg.ddpg import ddpg
import torch
import sys


number = 0
if len(sys.argv) == 2:
    print("Using model number: ",sys.argv[1])
    number = int(sys.argv[1])


actor_model =  None
critic_model =  None
actor_target_model = None
critic_target_model = None
actor_optimizer_model = None
critic_optimizer_model = None

imitation_actor_model = None #torch.load("net.torch_10000")


if number > 0:
    actor_model =  torch.load('ddpg_models/actor.torch_%d' %number)
    critic_model =  torch.load('ddpg_models/critic.torch_%d' %number)
    actor_target_model =  torch.load('ddpg_models/actor_target.torch_%d' %number)
    critic_target_model =  torch.load('ddpg_models/critic_target.torch_%d' %number)
    actor_optimizer_model =  torch.load('ddpg_models/actor_optimizer.torch_%d' %number)
    critic_optimizer_model =  torch.load('ddpg_models/critic_optimizer.torch_%d' %number)

ddpg(n_episodes=200,
        max_steps=1000,
        memory_size=50000,
        use_training_data=True,
        imitation_actor = imitation_actor_model,
        base_models=(actor_model, critic_model,
                    actor_target_model, critic_target_model,
                    actor_optimizer_model,critic_optimizer_model))

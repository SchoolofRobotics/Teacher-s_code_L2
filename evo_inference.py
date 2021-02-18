from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.registry import default_registry

import random
import numpy as np
import sys
from evolution import params_reshape, build_net, evaluate_population

l1_size = 16  # number of neurons in 1st layer
l2_size = 32  # number of neurons in 2nd layer
POP_SIZE = 12 # population size

if __name__ == "__main__":
    try:
        # This is a non-blocking call that only loads the environment.
        print ("Script started. Please start Unity environment to start training proccess.")
        engine_channel = EngineConfigurationChannel()
        # env = UnityEnvironment( side_channels=[engine_channel])
        env = default_registry["3DBall"].make(side_channels = [engine_channel])
        engine_channel.set_configuration_parameters(time_scale = 1, width=1920, height=1080) # control time scale 0.5 - half speed, 10. - 10x time
        # Start interacting with the environment.
        env.reset()
        # Info about our environment ---------------------
        print (f"number of behaviours: {len(list(env.behavior_specs) )}")
        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]
        action_spec = spec.action_spec
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        # Examine the number of observations per Agent
        print("Number of observations : ", len(spec.observation_shapes))
        print(" observations : ", spec.observation_shapes)
        # Is the Action continuous or multi-discrete ?
        if action_spec.is_continuous():
            print("The action is continuous")
            print(f"There are {action_spec.continuous_size} action(s)")
            # Create neural network structure and get its shape and flat params (weights, biases) for continues actions (-1;1)
            net_shapes, net_params = build_net(decision_steps.obs[0].shape[1],
                                 action_spec.continuous_size, l1_size, l2_size)

        # print (spec.action_spec.random_action() )
        if action_spec.is_discrete():
            print("The action is discrete")
            # How many actions are possible ?
            print(f"There are {action_spec.discrete_size} action(s)")
            for action, branch_size in enumerate(action_spec.discrete_branches):
                print(f"Action number {action} has {branch_size} different options")
            # Create neural network structure and get its shape and flat params (weights, biases) for continues actions (-1;1)
            net_shapes, net_params = build_net(decision_steps.obs[0].shape[1],
                                 action_spec.discrete_branches[0], l1_size, l2_size)
        
        for index, shape in enumerate(decision_steps.obs):
            # if len(shape) == 1:
            print(f"obs shape: {decision_steps.obs[index].shape}")
            print(f"First vector observations : {decision_steps.obs[index][0,:]} \n shape: {decision_steps.obs[index][0].shape}", )
        for index, shape in enumerate(terminal_steps.obs):
            # if len(shape) == 1:
            print(f"terminal obs shape: {terminal_steps.obs[index].shape}")
            print(f"First vector observations terminal : {terminal_steps.obs[index]}\
                    \n shape: {terminal_steps.obs[index].shape}", )
        # Info about our environment ---------------------
        param = np.loadtxt(sys.argv[1])
        # p = params_reshape(net_shapes, param)
        while True:
            ep_reward = 0
            pop = [param] * POP_SIZE
            episode_rewards = evaluate_population(pop, env, net_shapes, inference=True)
            print (f"average reward: {sum(episode_rewards)/len(episode_rewards)}")
    finally:
        env.close()
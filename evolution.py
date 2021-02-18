from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.registry import default_registry

import random
import numpy as np
import matplotlib.pyplot as plt
import os 

from deap import base, creator, tools

l1_size = 16  # number of neurons in 1st layer
l2_size = 32  # number of neurons in 2nd layer
NUM_EPISODES = 301 # number of episodes for training proccess
MIN_VALUE, MAX_VALUE = -1., 1. # min and max values for weights and biases so they don't go to infinity
POP_SIZE = 12 # population size
CXPB, MUTPB = 0.2, 0.1  # probabilities for crossover and mutation

def smooth_curve(points, factor=0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def params_reshape(net_shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(net_shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p

def sigmoid(x):
    """Sigmoid activation function"""
    return 1/(1+np.exp(-x))

def get_action(params, x, continuous=False):
    """
    params - weights and biases of single individual
    x - current observation for particular individual
    """
    x = x[np.newaxis, :] # reshaping input
    # Activation function for every layer - tanh. To use ReLU, use np.max(x, 0). For sigmoid, there is a function
    x = np.tanh(x.dot(params[0]) + params[1]) # Calculating 1st layer output
    x = np.tanh(x.dot(params[2]) + params[3]) # Calculating 2nd layer output
    if continuous: # for continuous action, negative means oposite direction action
        x = np.tanh(x.dot(params[4]) + params[5]) # Calculating output layer result (-1:1)
    else: # for discrete action
        x = sigmoid(x.dot(params[4]) + params[5]) # Calculating output layer result (0:1)
    return x[0]              

def build_net(s_dim, a_dim, layer1_size, layer2_size):
    """
    Build structure of neural network based on number of neurons in each layer. Returns weights and biases
    s_dim - number of observations (s - state) received.
    a_dim - number of total actions (a - action).
    layer1_size, layer2_size - number of neurons in hidden layers
    """
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(s_dim , layer1_size) # 1st layer weights and biases
    s1, p1 = linear(layer1_size, layer2_size) # 2nd layer weights and biases
    s2, p2 = linear(layer2_size, a_dim) # output layer weights and biases
    return [s0, s1, s2], np.concatenate((p0, p1, p2))

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

def initEA(icls, size, imin, imax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    return ind

def evaluate_population(pop, env, net_shapes, inference=False):
    """
    Evaluation of a whole population 
    """
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]
    action_spec = spec.action_spec

    fitnesses = [0.] * len(pop)
    env.reset()

    decision_steps, terminal_steps = env.get_steps(behavior_name)

    # Play in an environment for number of steps
    episode_length = 1000
    while episode_length > 1:
        for agent_id in terminal_steps.agent_id:
            fitnesses[agent_id] += terminal_steps[agent_id].reward
        
        if len(terminal_steps) == len(pop):
            break
        # Generate an action for all agents
        actions = np.empty( (len(decision_steps), action_spec.continuous_size) )
        for agent_id in decision_steps.agent_id:
            
            state =  decision_steps[agent_id].obs[0] 
            fitnesses[agent_id] += decision_steps[agent_id].reward
            individual = params_reshape(net_shapes, pop[agent_id])
            if action_spec.is_discrete():
                action_discrete_probs = get_action(individual, state) # returns probability for each action
                actions_discrete = np.argmax(action_discrete_probs) # choose action with highest probability
                actions[agent_id] = actions_discrete
            elif action_spec.is_continuous():
                action_continuous_probs = get_action(individual, state, continuous=True) # returns probability for each action
                actions[agent_id] = action_continuous_probs
        
        if episode_length % 251 == 0 and not inference:
            env.reset() # reset to change initial position from time to time
        else:
            if action_spec.is_discrete():
                action = ActionTuple(discrete=actions)
            elif action_spec.is_continuous():
                action = ActionTuple(continuous=actions)
            env.set_actions(behavior_name, action)
            env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        episode_length -= 1
    return fitnesses

def train( env, net_shapes, net_params):
    IND_SIZE = len(net_params.flat) # total number of parameters in neural network

    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # setting to maximaze fitness
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax) # setting that our individual is an array (weights and biases of neural net)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", initEA, creator.Individual, IND_SIZE, MIN_VALUE, MAX_VALUE,) # definition of individual
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # setting definition of population. In this case, list of individuals
    toolbox.register("mate", cxTwoPointCopy) # setting a function which defines crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0., sigma=1.0, indpb=0.1) # setting mutation type, indpb is probability for every value to mutate
    toolbox.register("select", tools.selTournament, tournsize=2) # setting selection method
    hof = tools.HallOfFame(3, similar=np.array_equal) # hall of fame to keep track of top 3 individuals

    pop = toolbox.population(n=POP_SIZE)
    fitnesses = evaluate_population(pop, env, net_shapes)
    print ("episode finished")
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)
    
    hof.update(pop)

    plot_mean = []
    plot_best = []

    for i in range(NUM_EPISODES):
        # Select the next generation individuals
        selected = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(ind) for ind in selected]
        # Shuffle before crossover 
        random.shuffle(offspring)
        # Apply crossover on the offspring based on probability CXPB . Delete fitness of children
        # -------------------------
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB: # only do crossover with probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # -------------------------

        # mutate individuals based on probability MUTPB. Delete fitness of the mutant
        # -------------------------
        for mutant in offspring:
            if random.random() < MUTPB: # only do mutation with probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # -------------------------

        # Evaluate the individuals
        fitnesses = evaluate_population(offspring, env, net_shapes)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = (fit,)

        pop[:] = offspring
        reward_mean = 0.0
        max_fit = -100000
        min_fit = 100000
        index = 0
        for j, ind in enumerate(pop):
            reward_mean = reward_mean + ind.fitness.values[0]
            if max_fit < ind.fitness.values[0]:
                max_fit = ind.fitness.values[0]
            if min_fit > ind.fitness.values[0]:
                min_fit = ind.fitness.values[0]
                index = j
        reward_mean = reward_mean / POP_SIZE
        #if top individual lost in next population, put it back instead of the weakest
        if max_fit < hof[0].fitness.values[0]:
            pop[index] = hof[0]

        hof.update(pop)

        if i%25 == 0 : # save top 3 individual params into file every 25 episodes
            np.savetxt('./results/individual1.txt', hof[0])
            np.savetxt('./results/individual2.txt', hof[1])
            np.savetxt('./results/individual3.txt', hof[2])
        plot_mean.append(reward_mean)
        plot_best.append(hof[0].fitness.values[0])
        
        print('| Generation: {:d} | Best individual Reward: {:f} | mean: {:f} '.format(i, hof[0].fitness.values[0],reward_mean))
    plot_mean = smooth_curve(plot_mean)
    # plot_best = smooth_curve(plot_best)
    plt.plot(range(1,NUM_EPISODES+1), plot_mean, 'b', label='mean reward')
    plt.plot(range(1,NUM_EPISODES+1), plot_best, 'r', label='best')
    plt.xlabel("episode")
    plt.ylabel("sum of rewards")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        if not os.path.isdir("results"):
            os.mkdir("results")
        # This is a non-blocking call that only loads the environment.
        print ("Script started. Please start Unity environment to start training proccess.")
        engine_channel = EngineConfigurationChannel()
        env = UnityEnvironment( side_channels=[engine_channel])
        # env = default_registry["3DBall"].make(side_channels = [engine_channel])
        engine_channel.set_configuration_parameters(time_scale = 10, width=1920, height=1080) # control time scale 0.5 - half speed, 10. - 10x time
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
            print(f"obs shape: {decision_steps.obs[index].shape}")
            print(f"First vector observations : {decision_steps.obs[index][0,:]} \n shape: {decision_steps.obs[index][0].shape}", )
        for index, shape in enumerate(terminal_steps.obs):
            print(f"terminal obs shape: {terminal_steps.obs[index].shape}")
            print(f"First vector observations terminal : {terminal_steps.obs[index]}\
                    \n shape: {terminal_steps.obs[index].shape}", )
        # Info about our environment ---------------------

        # set up evolution and start training
        train(env, net_shapes, net_params )
    finally:
        env.close()
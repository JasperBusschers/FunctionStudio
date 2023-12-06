import pygame
import numpy as np
import time
import torch
import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0

from src.functions import Function


def runEpisode(env, model):
    """
    Runs a single episode in the Gym environment using the given model as the policy.

    :param env: The Gym environment.
    :param model: The neural network model defining the policy.
    :return: The total reward obtained in the episode.
    """
    state,_ = env.reset()
    done = False
    total_reward1 = 0
    total_reward2 = 0
    i=0
    actionMapping = [(np.asarray([0, 0, 1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])) ,(np.asarray([
        0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))  ,(np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0]))
                     ,(np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1, 0, 0]))]
    while not done:
        state_tensor = torch.from_numpy(state).float()
        action = model.model(state_tensor).detach().numpy()
        # Map action to the environment's action space if necessary
        #action = actionMapping[np.argmax(action[0])]
        action =np.random.choice([0,1,2,3], 1, p=action)[0]
        #action = np.argmax(action)
        action=actionMapping[action]
        state, reward, done, _ = env.step(action)
        state=state[0]
        total_reward1 += reward[0]
        total_reward2 += reward[1]
        i+=1
    return total_reward1, total_reward2


env: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(
	max_steps=1000,
	render_treasure_values=True
)
# Create an instance of the GeneticOptimization class
optimization = Function(input_shape=11, output_shape=4)

# Add the runEpisode function as an objective
optimization.add_objective(lambda model: runEpisode(env, model), maximize=True, size = 2)

# Run the optimization process
weights, fintnesses = optimization.optimize(population_size=500, num_generations=100,tournament_size=5,
                                 cross_over_rate=0.8,
                      eta=0.4,stdev=0.8)
print(fintnesses)

# Close the environment
env.close()



import gym
import torch

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
    total_reward = 0
    action_counts = {0: 0, 1: 0}  # Track the number of times each action is taken
    i=0
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action = model.model(state_tensor).detach().numpy()
        # Map action to the environment's action space if necessary
        action = 0 if action[0][0] < 0.5 else 1

        state, reward, done, _,_ = env.step(action)
        total_reward += reward
        action_counts[action] += 1
        i+=1
        if i > 500:
            done = True


    return total_reward

# Assuming GeneticOptimization and NeuralNetwork classes are defined as provided earlier

# Create an instance of the Gym environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

# Create an instance of the GeneticOptimization class
optimization = Function(input_shape=4, output_shape=1)

# Add the runEpisode function as an objective
optimization.add_objective(lambda model: runEpisode(env, model), maximize=True)

# Run the optimization process
optimization.optimize(population_size=25, num_generations=100)

# Close the environment
env.close()

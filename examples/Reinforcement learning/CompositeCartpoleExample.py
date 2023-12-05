import gym
import torch

from src.CompositeFunction import CompositeFunction
from src.utils import pareto_filter


def runEpisode(env, compositeModel):
    """
    Runs a single episode in the Gym environment using the given model as the policy.

    :param env: The Gym environment.
    :param model: The neural network model defining the policy.
    :return: The total reward obtained in the episode.
    """
    state,_ = env.reset()
    done = False
    total_reward = 0
    i=0
    while not done:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action = compositeModel.infer({'sensor_processing' :state_tensor})['action_processing'].detach(
        ).numpy()
        # Map action to the environment's action space if necessary
        action = 0 if action[0][0] < 0.5 else 1

        state, reward, done, _,_ = env.step(action)
        total_reward += reward
        i+=1
        if i > 500:
            done = True
    return total_reward

# Assuming GeneticOptimization and NeuralNetwork classes are defined as provided earlier

# Create an instance of the Gym environment
env_name = 'CartPole-v1'
env = gym.make(env_name)


graph = {
    "sensor_processing": (4, 1),  # Node to process sensor inputs
    "decision_making": (1, 4),    # Node to make decisions based on processed inputs
    "action_processing": (4, 1)   # Node to process decisions into actions
}

edges = [
    ('sensor_processing', 'decision_making'),
    ('decision_making', 'action_processing')
]
composite_function = CompositeFunction(graph, edges)
# Add the runEpisode function as an objective to the 'cartpole_policy' node
composite_function.add_objective(lambda model: runEpisode(env,model), maximize=True)

# Run the optimization process for the composite function
weights, fitnesses = composite_function.optimize(population_size=25, num_generations=10)
#convert to numpy
fitnesses=fitnesses.numpy()
weights=weights.numpy()
#filter out only non dominated
non_dominated_ids = pareto_filter(fitnesses)
weights,fitnesses=[weights[i] for i in non_dominated_ids], [fitnesses[i] for i in non_dominated_ids]
# report on results
print(weights)
print(fitnesses)
# Close the environment
env.close()
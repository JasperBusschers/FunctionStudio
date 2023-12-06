import torch
from evotorch import Problem
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from evotorch.logging import StdOutLogger
import torch.nn as nn
from collections.abc import Iterable

from src.geneticAlgorithms import SteadyStateGAWithEarlyStopping
from src.neuralNetworkEstimator import NeuralNetwork


class Function:
    """
    Class to handle genetic optimization for a given problem, using objective functions.
    This class is designed to be agnostic of the specific details of the problem being optimized.
    """

    def __init__(self, input_shape, output_shape):
        """
        Initializes the genetic optimization class with a neural network model.

        :param input_shape: Integer, the size of the input layer of the neural network.
        :param output_shape: Integer, the size of the output layer of the neural network.
        """
        self.input_shape = input_shape
        self.model = NeuralNetwork(input_shape, output_shape)
        self.solution_length = self.model.get_solution_length()
        self.objectives = []
        self.maximizeOrMinimize = []

    def infer(self ,input_data ,weights=None):
        """
        Performs inference using the model with the given weights on the input data.

        :param input_data: The data on which inference is to be performed.
        :param weights: The weights to be set in the neural network model for inference.
        :return: The output of the model after performing inference.
        """
        # Set the provided weights to the model
        if weights is not None:
            self.model.set_weights(weights.clone())
        # Perform inference
        output = self.model.model(input_data)

        return output

    def set_weights(self,weights):
        if weights is not None:
            self.model.set_weights(weights.clone())

    def add_objective(self, objective_func, maximize=False, size=1):
        """
        Adds an objective function to the optimization process.

        :param objective_func: A callable that takes the model as an argument and returns a fitness score.
        """
        self.objectives.append(objective_func)
        if maximize:
            for i in range(size):
                self.maximizeOrMinimize.append("max")
        else:
            for i in range(size):
                self.maximizeOrMinimize.append("min")



    def evaluate_fitness(self, weights: torch.Tensor):
        """
        Evaluate the fitness of a solution based on the added objective functions.

        :param weights: Torch tensor, the weights to be set in the neural network model for evaluation.
        :return: Torch tensor, the computed fitness scores.
        """
        fitness_scores = [[] for o in self.maximizeOrMinimize]
        for x in weights:
            self.model.set_weights(x.clone())
            i=0
            for objective in self.objectives:
                score = objective(self.model)
                if isinstance(score, Iterable)and len(score)>1:
                    for s in score:
                        fitness_scores[i].append(s)
                        i += 1
                else:
                    fitness_scores[i].append(score)
                    i+=1
        if len(fitness_scores)>0:
            return   torch.stack([torch.tensor(f) for f in fitness_scores], dim=-1)
        else:
            return None

    def get_solution_length(self):
        """
        Returns the solution length of the neural network model.
        """
        return self.model.get_solution_length()

    def optimize(self, population_size=200, num_generations=100, tournament_size=2,cross_over_rate=0.95,eta=1,stdev=0.1):
        """
        Runs the genetic algorithm optimization.

        :param population_size: Integer, the size of the population for the genetic algorithm.
        :param num_generations: Integer, the number of generations for the genetic algorithm to run.
        """
        # Define the problem for the genetic algorithm
        problem = Problem(
            self.maximizeOrMinimize,  # Define the direction of optimization (minimization in this case)
            self.evaluate_fitness,             # Fitness evaluation function
            initial_bounds=(-1.0, 1.0),        # Initial bounds for the solutions
            solution_length=self.solution_length,
            vectorized=True
        )

        # Initialize the genetic algorithm
        ga = SteadyStateGAWithEarlyStopping(problem ,popsize=population_size)

        # Works like NSGA-II for multiple objectives
        ga.use(
            SimulatedBinaryCrossOver(problem ,tournament_size=tournament_size ,cross_over_rate=cross_over_rate ,
                                     eta=eta))
        ga.use(GaussianMutation(problem ,stdev=stdev))
        logger = StdOutLogger(ga)
        best ,steps = ga.run(num_generations,self.evaluate_fitness)
        resultingWeights = ga.population.values
        fitnesses = self.evaluate_fitness(resultingWeights)
        return ga.weights,ga.fitnesses




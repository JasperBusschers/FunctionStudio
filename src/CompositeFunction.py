from evotorch import Problem
from evotorch.algorithms import SteadyStateGA
from evotorch.logging import StdOutLogger
from evotorch.operators import SimulatedBinaryCrossOver ,GaussianMutation

import torch

from src.functions import Function


class CompositeFunction:
    def __init__(self, graph,edges):
        """
        Initializes a CompositeFunction with a given graph.

        :param graph: A dictionary representing the graph. Keys are node names,
                      and values are tuples (input_size, output_size).
        """
        self.edges = edges
        self.graph = graph
        self.nodes = {node: Function(input_size, output_size)
                      for node, (input_size, output_size) in graph.items()}
        self.outputs = {node: None for node in graph}  # To store outputs of each node
        self.objectives =[]
        self.maximizeOrMinimize=[]


    def run_local_optimization(self, node, population_size=100, num_generations=50):
        """
        Runs the optimization for a specific node.

        :param node: The name of the node.
        :param population_size: Population size for the genetic algorithm.
        :param num_generations: Number of generations for the genetic algorithm.
        """
        if node in self.nodes:
            self.outputs[node] = self.nodes[node].optimize(population_size, num_generations)
        else:
            raise ValueError(f"Node {node} not found in the graph.")

    def infer(self ,input_datas):
        """
        Executes the entire graph, ensuring data flows as per the graph's edges, with specified input data
        for root nodes.

        :param input_datas: A dictionary where keys are root node names and values are the input data for
        those nodes.
        """
        # Reset outputs
        self.outputs = {node:None for node in self.graph}

        # Set initial input data for root nodes
        for node ,data in input_datas.items():
            if node not in self.graph:
                raise ValueError(f"Node {node} not found in the graph.")
            self.outputs[node] = self.nodes[node].infer(data)

        # Perform a topological sort of nodes to respect dependencies
        sorted_nodes = self.topologicalOrder()
        # Traverse nodes in sorted order and perform inference
        for node in sorted_nodes:
            if self.outputs[node] is None:
                # Gather inputs from predecessor nodes
                input_data = [self.outputs[pred_node] for pred_node ,_ in self.edges if _ == node]

                # Perform inference if input data is available
                if all(data is not None for data in input_data):
                    # Concatenate inputs if there are multiple inputs
                    if len(input_data) > 1:
                        input_data = torch.cat(input_data ,dim=1)
                    else:
                        input_data = input_data[0]

                    self.outputs[node] = self.nodes[node].infer(input_data )
        return self.outputs

    def get_solution_length(self):
        """
        Returns the total solution length of all nodes in the composite structure.
        """
        total_length = 0
        for node in self.nodes.values():
            total_length += node.get_solution_length()
        return total_length


    def evaluate_fitness(self, weights: torch.Tensor):
        """
        Evaluates all the nodes in the composite function by splitting the weight vector
        according to the solution lengths of the nodes and using each node's evaluate_fitness function.

        :param weights: A single torch.Tensor representing the concatenated weights for all nodes.
        :return: A list of scores obtained by running the evaluate_fitness of the nodes.
        """
        scores = []
        start = 0
        for node_name, node in self.nodes.items():
                length = node.get_solution_length()
                node_weights = weights[start:start + length].unsqueeze(0)[0]  # Add batch
                # dimension
                # Use evaluate_fitness function of each GeneticOptimization node
                node_scores = node.evaluate_fitness(node_weights)
                if node_scores is not None or (node_scores and len(node_scores)==0):
                    # Assuming we're interested in the first objective's score
                    scores.append(node_scores.reshape(-1).numpy().tolist())

                start += length
        globalScores = []
        for w in weights:
            for o in self.objectives:
                self.set_weights(w)
                globalScores.append(o(self))
        scores.append(globalScores)
        return  torch.stack([torch.tensor(f) for f in scores], dim=-1)

    def set_weights(self,w):
        start=0
        for node_name, node in self.nodes.items():
                length = node.get_solution_length()
                node_weights = w[start:start + length]
                node.set_weights(node_weights)
                start += length
    def add_objective(self, objective_func, maximize=False):
        """
        Adds an objective function to the optimization process.

        :param objective_func: A callable that takes the model as an argument and returns a fitness score.
        """
        self.objectives.append(objective_func)
        if maximize:
            self.maximizeOrMinimize.append("max")
        else:
            self.maximizeOrMinimize.append("min")
    def topologicalOrder(self):
        """
        Determines the topological order of the nodes based on the directed edges.

        :return: A list of nodes in topological order.
        """
        # Create a dictionary to store the in-degree of each node
        in_degree = {node: 0 for node in self.graph}
        for source, destination in self.edges:
            in_degree[destination] += 1

        # Queue for nodes with in-degree 0
        queue = [node for node, degree in in_degree.items() if degree == 0]
        top_order = []

        while queue:
            node = queue.pop(0)
            top_order.append(node)

            # Decrease the in-degree of the neighboring nodes
            for _, destination in filter(lambda edge: edge[0] == node, self.edges):
                in_degree[destination] -= 1
                if in_degree[destination] == 0:
                    queue.append(destination)

        if len(top_order) != len(self.graph):
            raise Exception("Graph has a cycle, topological ordering is not possible")

        return top_order
    def optimize(self ,population_size=100 ,num_generations=50 ,tournament_size=2 ,cross_over_rate=0.95 ,
                 eta=1 ,stdev=0.1):
        """
        Optimizes the entire composite function using a single genetic algorithm.

        :param population_size: Integer, the size of the population for the genetic algorithm.
        :param num_generations: Integer, the number of generations for the genetic algorithm to run.
        """
        total_solution_length = self.get_solution_length()
        optimization_directions = []
        for node in self.nodes.values():
            optimization_directions.extend(node.maximizeOrMinimize)
        optimization_directions.extend(self.maximizeOrMinimize)
        # Define the problem for the genetic algorithm
        problem = Problem(
            optimization_directions ,  # Assume minimization for simplicity
            self.evaluate_fitness ,  # Fitness evaluation function
            initial_bounds=(-1.0 ,1.0) ,  # Initial bounds for the solutions
            solution_length=total_solution_length ,
            vectorized=True
        )

        # Initialize the genetic algorithm
        ga = SteadyStateGA(
            problem ,
            popsize=population_size ,
        )

        # Add operators to the GA (crossover, mutation, etc.)
        ga.use(SimulatedBinaryCrossOver(problem ,tournament_size=tournament_size ,
                                        cross_over_rate=cross_over_rate ,eta=eta))
        ga.use(GaussianMutation(problem ,stdev=stdev))

        # Optionally, add a logger
        logger = StdOutLogger(ga)

        # Run the genetic algorithm
        best_solution = ga.run(num_generations)

        resultingWeights = ga.population.values
        fitnesses = self.evaluate_fitness(resultingWeights)
        return resultingWeights,fitnesses


import torch

from src.functions import Function


def supervised_learning_objective(model, X, Y):
    """
    Objective function for Supervised learning - minimizes prediction error.

    :param model: The neural network model.
    :param X: Input features for the model.
    :param Y: Actual target values (prices).
    :return: Loss (error) value.
    """
    predictions = model.model(X)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(predictions, Y)
    return loss


def constraint_objective(model, X, min_price, max_price):
    """
    Objective function to enforce constraints, e.g., prices within a certain range.

    :param model: The neural network model.
    :param X: Input features for the model.
    :param min_price: Minimum allowable price.
    :param max_price: Maximum allowable price.
    :return: Penalty score for violating constraints.
    """
    predictions = model.model(X)
    penalty = torch.where(predictions < min_price, min_price - predictions, 0) + \
              torch.where(predictions > max_price, predictions - max_price, 0)
    return torch.sum(penalty)


def margin_objective(model, X, cost):
    """
    Objective function to maximize profit margins.

    :param model: The neural network model.
    :param X: Input features for the model.
    :param cost: Cost associated with the product.
    :return: Negative margin (since we minimize in the genetic algorithm).
    """
    predictions = model.model(X)
    margin = predictions - cost
    return -torch.mean(margin)


# Assuming GeneticOptimization and NeuralNetwork classes are defined as provided earlier

# Example data (placeholders, replace with actual data)
X_train = torch.randn(100, 2)  # Example feature data
Y_train = torch.randn(100, 1)  # Example target prices
min_price = torch.tensor([0.5])  # Minimum price
max_price = torch.tensor([1.5])  # Maximum price
cost = torch.tensor([0.3])  # Cost of the product

# Create an instance of the GeneticOptimization class
optimization = Function(input_shape=2, output_shape=1)

# Add the objective functions
optimization.add_objective(lambda model: supervised_learning_objective(model, X_train, Y_train))
optimization.add_objective(lambda model: constraint_objective(model, X_train, min_price, max_price))
optimization.add_objective(lambda model: margin_objective(model, X_train, cost))

# Run the optimization process
optimization.optimize(population_size=100, num_generations=50)
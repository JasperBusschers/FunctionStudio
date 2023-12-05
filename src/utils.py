
import numpy as np

def pareto_filter(costs, minimize=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    from https://stackoverflow.com/a/40239615
    """
    costs_copy = np.copy(costs) if minimize else -np.copy(costs)
    is_efficient = np.arange(costs_copy.shape[0])
    n_points = costs_copy.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs_copy):
        nondominated_point_mask = np.any(
            costs_copy < costs_copy[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Remove dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs_copy = costs_copy[nondominated_point_mask]
        next_point_index = np.sum(
            nondominated_point_mask[:next_point_index]) + 1
    return [i for i in is_efficient]

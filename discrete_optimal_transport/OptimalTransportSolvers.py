import numpy as np
from abc import ABC, abstractmethod

from scipy.stats import norm
from scipy.spatial import distance
import numpy as np
from scipy.optimize import linear_sum_assignment

class OptimalTransportProblem(ABC):

    def __init__(self, 
                 kwargs_source : dict, 
                 kwargs_destination : dict, 
                 n_samples : int) -> None:

        self.source_coordinates = [tuple(norm.rvs(**kwargs_source)) for i in range(0, n_samples)]
        self.destination_coordinates = [tuple(norm.rvs(**kwargs_destination)) for i in range(0, n_samples)]
    
    @abstractmethod
    def solve(self) -> float:
        pass

class HungarianSolver(OptimalTransportProblem):
    
    def __init__(self, 
                 kwargs_source : dict, 
                 kwargs_destination : dict, 
                 n_samples : int) -> None:
        
        super().__init__(kwargs_source, kwargs_destination, n_samples)

    def solve(self) -> float:

        cost_matrix = distance.cdist(self.source_coordinates, self.destination_coordinates, 'euclidean')
        x_index, y_index = linear_sum_assignment(cost_matrix)
        
        wasserstein_metric = cost_matrix[x_index, y_index].sum()

        return wasserstein_metric

        
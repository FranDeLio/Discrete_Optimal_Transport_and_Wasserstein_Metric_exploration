import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.spatial import distance
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ortools.graph.python import min_cost_flow
import pyomo.environ as pe
import pyomo.opt as po

import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class OptimalTransportProblem(ABC):

    def __init__(self, 
                 kwargs_source : dict, 
                 kwargs_destination : dict, 
                 n_samples : int) -> None:

        self.n_samples = n_samples
        self.source_coordinates = [tuple(norm.rvs(**kwargs_source)) for i in range(0, self.n_samples)]
        self.destination_coordinates = [tuple(norm.rvs(**kwargs_destination)) for i in range(0, self.n_samples)]
        self.cost_matrix = distance.cdist(self.source_coordinates, self.destination_coordinates, 'euclidean')
    
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

        x_index, y_index = linear_sum_assignment(self.cost_matrix)
        
        wasserstein_metric = self.cost_matrix[x_index, y_index].sum()

        return wasserstein_metric
    

class MinCostFlowSolver(OptimalTransportProblem):
    
    def __init__(self, 
                 kwargs_source : dict, 
                 kwargs_destination : dict, 
                 n_samples : int) -> None:
        
        super().__init__(kwargs_source, kwargs_destination, n_samples)

    def solve(self) -> float:

        model = min_cost_flow.SimpleMinCostFlow()

        # connect source node to first layer nodes
        for i in range(1, 1+self.n_samples):
            model.add_arc_with_capacity_and_unit_cost(0, i, 1, 0)
    
        # connect first layer of nodes to the second layer
        for i in range(1, 1+self.n_samples):
            for j in range(1+self.n_samples, 1+self.n_samples+self.n_samples):
                model.add_arc_with_capacity_and_unit_cost(i, j, 1, int(self.cost_matrix[i-1, j-self.n_samples-1]))

        # connect the second layer to the delivery node
        for i in range(1+self.n_samples, 1+self.n_samples+self.n_samples):
                model.add_arc_with_capacity_and_unit_cost(i, 1+self.n_samples+self.n_samples, 1, 0)

        # lastly we specify the delivery node requests all original samples and all original samples need to be delivered
        for i in range(0, 2+self.n_samples+self.n_samples):

            if i==0:
                model.set_node_supply(i, self.n_samples)
            
            elif i==1+self.n_samples+self.n_samples:
                model.set_node_supply(i, -self.n_samples)

            else:
                model.set_node_supply(i, 0)

        model.solve()

        wasserstein_metric = model.optimal_cost()
                
        return wasserstein_metric


class MILPSolver(OptimalTransportProblem):
    
    def __init__(self, 
                 kwargs_source : dict, 
                 kwargs_destination : dict, 
                 n_samples : int) -> None:
        
        super().__init__(kwargs_source, kwargs_destination, n_samples)

    def solve(self, 
              decision_variable_type : str = 'real',
              solver_name : str = 'cbc') -> float:

        if decision_variable_type == 'real':
            decision_domain = pe.NonNegativeReals
        else:
            decision_domain = pe.Binary

        solver = po.SolverFactory(solver_name)
        model = pe.ConcreteModel()

        model.source = pe.Set(initialize=set(range(0, self.n_samples)))
        model.destination = pe.Set(initialize=set(range(0, self.n_samples)))

        cost = pd.DataFrame(self.cost_matrix).stack().reset_index().set_index(['level_0','level_1'])[0].to_dict()
        model.cost = pe.Param(model.source, model.destination, initialize=cost)

        model.y = pe.Var(model.source, model.destination, domain=decision_domain, initialize=0) # variable: 1 if assign parameters set k to city c else 0.

        expression = sum(model.cost[c, k] * model.y[c, k] for c in model.source for k in model.destination)
        model.obj = pe.Objective(sense=pe.minimize, expr=expression)

        def all_served(model, k):
            # assign exactly one parameter set k to every city c.
            constraint = (sum(model.y[c, k] for c in model.source) == 1)
            return constraint

        def city_unicity(model, c):
            # assign exactly one parameter set k to every city c.
            constraint = (sum(model.y[c, k] for k in model.destination) == 1)
            return constraint

        model.all_served = pe.Constraint(model.destination, rule=all_served)
        model.con_unicity = pe.Constraint(model.source, rule=city_unicity)

        result = solver.solve(model, timelimit=60)

        wasserstein_metric = result['Problem'][0]['Lower bound']

        return wasserstein_metric






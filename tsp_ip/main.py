import pulp as pl
import numpy as np
import networkx as nx
from typing import Dict, Tuple


def tsp_ip(init: nx.DiGraph | np.ndarray, msg: bool = False) -> (int | float | None, nx.DiGraph):
    """
    Solves the asymmetric Traveling Salesman Problem using integer linear programming.

    Parameters:
        init_graph (nx.DiGraph): The directed graph representing the distances
        or adjacency matrix numpy.

    Returns:
        Length of the shortest path,
        nx.DiGraph: The optimized route as a directed graph.
    """
    
    # Init griph
    if isinstance(init, nx.DiGraph):        
        init_graph = init.copy()
    else:
        init_graph = nx.from_numpy_array(init, create_using=nx.DiGraph)

    # Number of graph vertices
    n = len(init_graph)
    if n <= 1:
        return None, nx.DiGraph()
        
    # Removing self-loops
    init_graph.remove_edges_from(nx.selfloop_edges(init_graph))    
    
    # Fill index dictionaries
    array_index: Dict[int, Tuple[int, int]] = {}
    array_key: Dict[Tuple[int, int], int] = {}
    for i, key in enumerate(init_graph.in_edges()):
        array_key[key] = i
        array_index[i] = key

    # Declare the model
    model = pl.LpProblem(name="tsp", sense=pl.LpMinimize)

    # Connect the solver
    solver = pl.PULP_CBC_CMD(msg=msg)

    # Declare model variables
    x = [pl.LpVariable(name=f'x_{array_index[i][0]}_{array_index[i][1]}', cat='Integer', lowBound=0) for i in array_index]

    # Input the objective function
    model += pl.lpSum([init_graph[array_index[i][0]][array_index[i][1]]['weight'] * x[i] for i in array_index])

    # Set initial constraints
    for i in range(n):
        # Each node can have only one outgoing edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.out_edges(i)]) == 1
        # Each node can have only one incoming edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.in_edges(i)]) == 1

    half_n = n // 2
    while True:
        # Find a solution
        status = model.solve(solver)        

        if status != 1:
            return None, nx.DiGraph()

        # Save solver result as a graph
        graph_result = nx.DiGraph()
        for val in model.variables():
            if round(val.value()) == 1:
                key = val.name.split('_')
                graph_result.add_edge(int(key[1]), int(key[2]))

        # Split the result graph into subsets
        result_sets = list(nx.connected_components(graph_result.to_undirected()))

        # Solution found if there is only one subset in the graph
        if len(result_sets) == 1:
            return sum([init_graph[i[0]][i[1]]['weight'] for i in graph_result.edges()]), graph_result

        # For each subset, add a constraint connecting it to other subsets
        for val in result_sets:
            if len(val) <= half_n:
                model += pl.lpSum([
                    x[array_key[key]] for key in init_graph.out_edges(val) if key[0] in val and key[1] in val
                ]) <= len(val) - 1


def tsp_mtz(init: nx.DiGraph | np.ndarray, msg: bool = False) -> (int | float | None, nx.DiGraph):
    """
    Solves the asymmetric Traveling Salesman Problem using integer linear programming.
    In Miller–Tucker–Zemlin formulation

    Parameters:
        init_graph (nx.DiGraph): The directed graph representing the distances
        or adjacency matrix numpy.

    Returns:
        Length of the shortest path,
        nx.DiGraph: The optimized route as a directed graph.
    """
   
    # Init griph
    if isinstance(init, nx.DiGraph):        
        init_graph = init.copy()
    else:
        init_graph = nx.from_numpy_array(init, create_using=nx.DiGraph)

    # Number of graph vertices
    n = len(init_graph)
    if n <= 1:
        return None, nx.DiGraph()
        
    # Removing self-loops
    init_graph.remove_edges_from(nx.selfloop_edges(init_graph))  
       
    # Fill index dictionaries
    array_index = {}
    array_key = {} 
    for i, key in enumerate(init_graph.in_edges()):
        array_key[key] = i
        array_index[i] = key

    # Declare the model
    model = pl.LpProblem(name="tsp", sense=pl.LpMinimize)
    
    # Connect the solver
    solver = pl.PULP_CBC_CMD(msg=msg)
    
    # Declare model variables
    u = [pl.LpVariable(name=f'u_{i}', cat='Integer', lowBound=0) for i in range(n - 1)]
    x = [pl.LpVariable(name=f'x_{array_index[i][0]}_{array_index[i][1]}', cat='Integer', lowBound=0) for i in array_index]
           
    # Input the objective function
    model += pl.lpSum([init_graph[array_index[i][0]][array_index[i][1]]['weight'] * x[i] for i in array_index])
    
    for i in range(1, n):
        for j in range(1, n):
            if i != j and (i, j) in array_key:
                model += u[i - 1] - u[j - 1] + n * x[array_key[(i, j)]] <= n - 1    
    
    # Set initial constraints
    for i in range(n):
        # Each node can have only one outgoing edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.out_edges(i)]) == 1
        # Each node can have only one incoming edge
        model += pl.lpSum([x[array_key[j]] for j in init_graph.in_edges(i)]) == 1

    # Find a solution
    status = model.solve(solver)

    if status != 1:
        return None, nx.DiGraph()

    # Save solver result as a graph
    graph_result = nx.DiGraph()
    for val in model.variables():
        key = val.name.split('_')
        if round(val.value()) == 1 and key[0] == 'x':            
            graph_result.add_edge(int(key[1]), int(key[2]))            
    return sum([init_graph[i[0]][i[1]]['weight'] for i in graph_result.edges()]), graph_result

# tsp-ip
TSP-IP (Travelling Salesman Problem integer programing)

Installing the package

	pip install tsp-ip

Example 1

```python
import numpy as np
from tsp_ip import tsp_ip

n = 20
print('Graph size = ', n)
matrix = np.random.randint(1, 99, (n, n))
print(matrix)

# Find the solution init as matrix numpy
path_len, graph_result = tsp_ip(matrix)

if path_len:
    # Get the list of edges of the optimal path
    print('Min path =', nx.find_cycle(graph_result))
    print('Length =', path_len)
else:
    print('Solution impossible!')
```


Example 2

```python
from tsp_ip import tsp_ip
from math import hypot
import random as rnd
import networkx as nx
import matplotlib.pyplot as plt

def distance(point1, point2):
    return hypot(point2[0] - point1[0], point2[1] - point1[1])

n = 50
print('Graph size = ', n)
points = [(rnd.uniform(0, 1000), rnd.uniform(0, 1000)) for _ in range(n)]

init_graph = nx.DiGraph()
for i in range(n):
    for j in range(n):
        if j < i:
            length = round(distance(points[i], points[j]))
            init_graph.add_edge(i, j, weight=length)
            init_graph.add_edge(j, i, weight=length)

# Find the solution init as graph
path_len, graph_result = tsp_ip(init_graph, msg=True)

if path_len:
    # Get the list of edges of the optimal path
    print('Min path =', nx.find_cycle(graph_result))
    print('Length =', path_len)
else:
    print('Solution impossible!')

# Draw the graph
plt.figure(figsize=(10, 8))
plt.axis("equal")
nx.draw(graph_result, points, width=0, with_labels=True, node_size=0, font_size=9, font_color="blue", arrowsize=0.1)
nx.draw(graph_result, points, width=1, edge_color="red", style="-", with_labels=False, node_size=0, arrowsize=10)
plt.show()
```


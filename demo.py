from graph import *
import time
from utils import *


nfile = 'killian-v.dat'
efile = 'killian-e.dat'

nodes, edges = readData(nfile, efile)

# nodes[i] = [x, y, heading]
# edges[i] = [index of related node, mean, infm]

graph = Graph()

n = nodes.shape[0]

for i in range(n):
    start = time.time()
    graph.addNode(i, nodes[i])

    for j in range(len(edges[i])):
        idx_j = i
        idx_i = edges[i][j][0]
        mean = edges[i][j][1]
        infm = edges[i][j][2]

        graph.addEdge(idx_j, idx_i, mean, infm)

plotGraph(graph, 'bo')
graph.optimize(3)
plotGraph(graph, 'ro')


#!/usr/bin/env python
"""
This example introduces the concept of evolution using the simulated annealing
metaheuristic. The user must provide functions that quantify a performance
measure (smaller is better - like a cost) and a mutation that should be 
performed to search for new network topologies. The performance function used
here is based on the topological eigenratio measure that is calculated from
the network topology and does not require simulation of the dynamics. Networks
with reduced eigneratios are known to show improved synchronisation.
"""

#=========================================
# IMPORT THE LIBRARIES
#=========================================

# So that we can run the examples without netevo necessarily being 
# in the system path.
import sys
sys.path.append('../netevo')
import netevo
import random
import networkx as nx
import numpy as np

#=========================================
# DEFINE MUTATION FUNCTION
#=========================================

# Define a function for searching for new networks (random rewire)
def rewire (G):
    n_to_rewire = int(random.expovariate(4.0))
    if n_to_rewire < 1:
        n_to_rewire = 1
    netevo.random_rewire (G, n_to_rewire)

#=========================================
# DEFINE PERFORMANCE MEASURE
#=========================================

# Define a function for the performance measure (eigenratio: smaller = better)
def eigenratio (G):
    if nx.is_connected(G):
        # Calculate the eigenrato (lambda_N/lambda_2)
        L = nx.laplacian_matrix(G)
        eigenvalues, eigenvectors = np.linalg.eig(L)
        idx = eigenvalues.argsort()   
        eigenvalues = eigenvalues[idx]
        return eigenvalues[-1] / eigenvalues[1]
    else:
        # If the network is not connected it is not valid
        return float('inf')

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a random undirected graph and check valid (n=50, m=100)
n = 50
G = []
while True:
    G = nx.gnm_random_graph(n, 2*n)
    if eigenratio(G) != float('inf'):
        break

# No dynamics are required
G.graph['node_dyn'] = False
G.graph['edge_dyn'] = False

#=========================================
# EVOLVE THE NETWORK
#=========================================

# Perform the evolution
iteration, G_final = netevo.evolve_sa(G, eigenratio, rewire, 
                                      initial_temp=100.0, min_temp=0.0000001, 
                                      reporter=netevo.evo_sa_reporter)

# Output GML files containing the initial and final toplogies
netevo.write_to_file(G, 'evolution_sa_initial.gml')
netevo.write_to_file(G_final, 'evolution_sa_final.gml')

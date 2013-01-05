#!/usr/bin/python
"""
evolution_sa.py

This example introduces the concept of evolution using a simulated annealing
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
sys.path.append('../netevo-py')
import netevo
import math
import networkx as nx
import random
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

#=========================================
# DEFINE MUTATIONS
#=========================================

# Define a function for searching for new networks
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
		eigenvalues, eigenvectors = linalg.eig(L)
		idx = eigenvalues.argsort()   
		eigenvalues = eigenvalues[idx]
		return eigenvalues[-1]/eigenvalues[1]
	# The network is not connected -  it is not valid
	return float('inf')

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a random graph, but check that it is valid
n = 100
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
iteration, G_final = netevo.evolve_sa (G, eigenratio, rewire, initial_temp=100.0, min_temp=0.0000001)

# Output GML files containing the initial and final toplogies (viewable in Cytoscape/yEd)
netevo.write_gml(G, 'evolution_sa_initial.gml')
netevo.write_gml(G_final, 'evolution_sa_final.gml')

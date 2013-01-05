#!/usr/bin/python
"""
evolution_sa.py

Example
"""

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


def rewire (G):
	n_to_rewire = int(random.expovariate(4.0))
	if n_to_rewire < 1:
		n_to_rewire = 1
	netevo.random_rewire (G, n_to_rewire)


def eigenratio (G):
	if nx.is_connected(G):
		L = nx.laplacian_matrix(G)
		eigenvalues, eigenvectors = linalg.eig(L)
		idx = eigenvalues.argsort()   
		eigenvalues = eigenvalues[idx]
		return eigenvalues[-1]/eigenvalues[1]

	return float('inf')


# Create a random graph, but check that it is valid
n = 100
G = []
while True:
	G = nx.gnm_random_graph(n, 2*n)
	if eigenratio(G) != float('inf'):
		break
	
G.graph['node_dyn'] = False
G.graph['edge_dyn'] = False

iteration, G_final = netevo.evolve_sa (G, eigenratio, rewire, initial_temp=100.0, min_temp=0.0000001)

netevo.write_gml(G, 'out_start.gml')
netevo.write_gml(G_final, 'out_end.gml')

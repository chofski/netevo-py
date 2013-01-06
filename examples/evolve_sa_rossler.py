#!/usr/bin/python
"""
evolution_sa_.py

NetEvo example that introduces the concept of evolution using a simulated annealing
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

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Define a function for the continuous node dynamics
def rossler_node_dyn (G, n, t, state):	
	# Parameters
	coupling = 0.5

	# Calculate the new state value
	c1 = 0.0
	c3 = 0.0
	
	for i in G[n]:
		c1 += coupling * (G.node[i]['state'][0] - state[0])
		c3 += coupling * (G.node[i]['state'][2] - state[2])
	
	v1 = -state[1] - state[2] + c1   
	v2 = state[0] + 0.165 * state[1]
	v3 = 0.2 + (state[0] - 10.0) * state[2] + c3 
	
	# Return the new state value for the node
	return np.array([v1, v2, v3])

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

# Define a function for the performance measure (order_parameter: smaller = better)
def order_parameter (G):
	if nx.is_connected(G):
		# Simulate from random initial conditions
		netevo.rnd_uniform_node_states(G, [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)])
		print 'here'
		netevo.simulate_rk45(G, 50.0, netevo.no_state_reporter, h=0.1)
		print 'and here'
		# Calculate the order_parameter and return value
		mu = 0.0
		for i in G.nodes():
			for j in G.nodes():
				if i != j:
					dist = np.linalg.norm(G.node[i]['state']-G.node[j]['state'])
					# Heaviside function (allow for some numerical error: 0.01)
					if dist - 0.01 >= 0.0:
						mu += 100.0
		return mu * (1.0 / (G.number_of_nodes() * (G.number_of_nodes() - 1.0)));
	# If the network is not connected it is not valid
	return float('inf')

#=========================================
# CREATE THE NETWORK
#=========================================

# Create a random undirected graph and check valid (n=50, m=100)
n = 25
G = []
while True:
	G = nx.gnm_random_graph(n, 2*n)
	if nx.is_connected(G):
		break

# No dynamics are required
G.graph['node_dyn'] = True
G.graph['edge_dyn'] = False

netevo.set_all_node_dynamics(G, rossler_node_dyn)

#=========================================
# EVOLVE THE NETWORK
#=========================================

# Perform the evolution (using simulated dynamics as part of the performance measure)
iteration, G_final = netevo.evolve_sa (G, order_parameter, rewire, initial_temp=100.0, min_temp=0.00001)

# Output GML files containing the initial and final toplogies (viewable in Cytoscape/yEd)
netevo.write_to_file(G, 'evolution_sa_dyn_initial.gml')
netevo.write_to_file(G_final, 'evolution_sa_dyn_final.gml', node_keys=['state', 'new_state'])

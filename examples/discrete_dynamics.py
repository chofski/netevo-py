#!/usr/bin/python
"""
discrete_dynamics.py

NetEvo example showing how a discrete node dynamics can be simulated. A ring
network of identical kuramoto nodes is created and each node given a random
initial state. Nodes are diffusely coupled along the edges and simulation of
the network sees a fast convergence to a synchronised state. Output from the
simulation is printed to the screen.
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

#=========================================
# DEFINE THE DYNAMICS
#=========================================

# Define a function for the discrete node dynamics
def kuramoto_node_dyn (G, n, t, state):
	# Parameters
	cur_params = G.node[n]['params']
	natural_freq = cur_params[0]
	coupling_strength = cur_params[1]
	
	# Calculate the new state value
	sum_coupling = 0.0
	for i in G[n]:
		sum_coupling += math.sin(G.node[i]['state'] - state)
		
	# Calcuate the new state of the node and return the value
	return math.fmod(state + natural_freq + (coupling_strength * sum_coupling), 6.283)

#=========================================
# CREATE THE DYNAMICAL NETWORK
#=========================================

# Create an empty graph (undirected)
G2 = nx.Graph()

# We only need node dynamics
G2.graph['node_dyn'] = True
G2.graph['edge_dyn'] = False

# Create the network of n nodes connected in a ring
n_nodes = 4
G2.add_node(0)
G2.node[0]['dyn'] = kuramoto_node_dyn
G2.node[0]['params'] = [0.2, 0.1]
for i in range(1, n_nodes):
	# Create the node
	G2.add_node(i)
	# Set the dynamics of the new node
	G2.node[i]['dyn'] = kuramoto_node_dyn
	# All nodes have identical dynamical parameters
	G2.node[i]['params'] = [0.2, 0.1]
	# Connect it to the previous node in the ring
	G2.add_edge(i-1,i)

# Set the initial state to a random number in the range (0, 6.0)
netevo.rnd_uniform_node_states (G2, [(0.0, 6.0)])

#=========================================
# SIMULATE THE NETWORK DYNAMICS
#=========================================

# Simulate the dynamics and print the state to the screen
netevo.simulate_steps (G2, 100, netevo.state_reporter)
